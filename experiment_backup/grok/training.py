#!/usr/bin/env python

import argparse
import copy
import json
import logging
import math
import os
import sys
import pickle
from argparse import ArgumentParser, Namespace
from functools import reduce
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
import time


class HParamsDict(dict):
    """字典类，支持点号访问（用于 PyTorch Lightning 兼容性）"""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import grok.metrics as metrics
from grok.data import (
    DEFAULT_DATA_DIR,
    EOS_TOKEN,
    VALID_OPERATORS,
    ArithmeticDataset,
    ArithmeticIterator,
)
from grok.transformer import Transformer
from grok.measure import get_sharpness

DEFAULT_LOG_DIR = "logs"


class CustomEarlyStopping(Callback):
    """自定义早停Callback，基于验证准确率阈值和延迟步数"""
    def __init__(self, threshold: float = 0, delay_steps: int = 2000):
        super().__init__()
        self.threshold = threshold
        self.delay_steps = delay_steps
        self._trigger_step = None
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """在验证epoch结束时检查是否触发早停"""
        if self.threshold <= 0:
            return
        
        # 获取验证准确率
        if not hasattr(pl_module, 'last_val_accuracy'):
            return
        
        acc_value = pl_module.last_val_accuracy
        if acc_value >= self.threshold:
            if self._trigger_step is None:
                self._trigger_step = trainer.global_step
                print(f"[Early Stop] 验证准确率达到 {acc_value:.2f}% (阈值: {self.threshold}%)，将在 {self.delay_steps} 步后停止")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """在每个训练batch结束时检查是否应该停止"""
        if self._trigger_step is not None:
            steps_since_trigger = trainer.global_step - self._trigger_step
            if steps_since_trigger >= self.delay_steps:
                print(f"[Early Stop] 在训练步 {trainer.global_step} 停止训练（已达到延迟步数 {self.delay_steps}，实际已过 {steps_since_trigger} 步）")
                # 使用 should_stop 标志停止训练（兼容 PyTorch Lightning 2.x）
                trainer.should_stop = True


class TrainableTransformer(LightningModule):
    """
    Adds training methods to train a generic transformer on arithmetic equations
    """

    def __init__(self, hparams: Namespace) -> None:
        """
        :param hparams: An argparse.Namespace with parameters defined in
                        self.add_model_specific_args().
        """
        super().__init__()
        # Convert Namespace to dict for both Lightning and our code
        if isinstance(hparams, Namespace):
            hparams_dict = vars(hparams)
        elif isinstance(hparams, dict):
            hparams_dict = hparams
        else:
            hparams_dict = vars(hparams)
        
        # Save hyperparameters (Lightning 2.6+ compatibility)
        self.save_hyperparameters(hparams_dict)
        
        # Replace Lightning's hparams with HParamsDict that supports dot notation
        # This must be done after save_hyperparameters to avoid breaking Lightning's internal calls
        if hasattr(self, '_hparams'):
            # Convert Lightning's hparams dict to HParamsDict
            original_hparams = self._hparams
            self._hparams = HParamsDict(original_hparams if isinstance(original_hparams, dict) else dict(original_hparams))
        
        self.prepare_data()

        self.transformer = Transformer(
            hparams.n_layers,
            hparams.n_heads,
            hparams.d_model,
            hparams.dropout,
            hparams.max_context_len,
            len(self.train_dataset.tokenizer),
            hparams.non_linearity,
            weight_noise=self.hparams.weight_noise,
        )

        # 应用初始化微扰（如果启用）
        perturbation_seed = getattr(hparams, "perturbation_seed", None)
        perturbation_scale = getattr(hparams, "perturbation_scale", 0.001)
        if perturbation_seed is not None:
            self._apply_init_perturbation(perturbation_seed, perturbation_scale)

        self.margin = torch.Tensor([0])
        self.next_epoch_to_eval = -1
        self.next_train_epoch_to_log = 0
        # 每个epoch都记录训练指标，而不是按指数增长
        # Store step outputs for epoch end hooks (PyTorch Lightning 2.0+)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Fingerprint 相关初始化
        self.enable_fingerprint = getattr(hparams, "enable_fingerprint", False)
        if self.enable_fingerprint:
            import grok.fingerprint as fingerprint

            # 创建 fingerprint 目录
            self.fingerprint_dir = os.path.join(hparams.logdir, "fingerprints")  # type: ignore
            os.makedirs(self.fingerprint_dir, exist_ok=True)

            # 获取 fingerprint 来源模式（hidden_state 或 log_prob）
            self.fingerprint_source = getattr(hparams, "fingerprint_source", "hidden_state")
            
            # 根据来源确定嵌入方式
            if self.fingerprint_source == "log_prob":
                # log_prob 模式：使用标量分位数嵌入（正确答案的 log-prob）
                # 输入: M 个标量 -> 输出: Q 维分位点向量
                self.quantile_embedder = fingerprint.ScalarQuantileEmbedding(
                    n_quantiles=getattr(hparams, "n_quantiles", 19),
                )
                embed_dim = 1  # 一维标量分布
                n_projections = 0  # 不需要投影
            else:
                # hidden_state 模式：使用随机投影 + 分位数嵌入
                embed_dim = hparams.d_model  # type: ignore
                n_projections = getattr(hparams, "proj_dim", 100)
                self.quantile_embedder = fingerprint.QuantileEmbedding(
                    d_model=embed_dim,
                    n_projections=n_projections,
                    n_quantiles=getattr(hparams, "n_quantiles", 19),
                    random_seed=getattr(hparams, "fingerprint_seed", 42),
                )

            # 保存配置
            config = {
                "d_model": embed_dim,
                "n_projections": n_projections,
                "n_quantiles": getattr(hparams, "n_quantiles", 19),
                "probe_size": getattr(hparams, "probe_size", 100),
                "record_every": getattr(hparams, "record_every", 100),
                "layer_name": getattr(hparams, "layer_name", "decoder.blocks"),
                "token_index": getattr(hparams, "token_index", -1),
                "center_mode": getattr(hparams, "center_mode", "per_snapshot_mean"),
                "fingerprint_seed": getattr(hparams, "fingerprint_seed", 42),
                "fingerprint_source": self.fingerprint_source,
            }
            fingerprint.save_fingerprint_config(self.fingerprint_dir, config)
            
            # 只在 hidden_state 模式下保存投影方向
            if self.fingerprint_source != "log_prob":
                fingerprint.save_projection_directions(
                    self.fingerprint_dir, self.quantile_embedder.projection_directions
                )

            # 选择 probe 样本（固定随机种子）
            probe_seed = getattr(hparams, "fingerprint_seed", 42)
            probe_indices = fingerprint.select_probe_samples(
                self, getattr(hparams, "probe_size", 100), random_seed=probe_seed
            )
            self.probe_indices = probe_indices
            fingerprint.save_probe_indices(
                self.fingerprint_dir, probe_indices.cpu().numpy()
            )

            # 存储激活值的 hook
            self.activation_hook = None
            self.captured_activation = None

            # 初始化参考快照（用于 relative_to_t0 模式）
            self.y_0_ref = None

            # 记录参数
            self.record_every = getattr(hparams, "record_every", 100)
            self.layer_name = getattr(hparams, "layer_name", "decoder.blocks")
            self.token_index = getattr(hparams, "token_index", -1)
            self.center_mode = getattr(hparams, "center_mode", "per_snapshot_mean")

            # 注册 forward hook 来捕获激活值
            self._register_activation_hook()
        else:
            self.fingerprint_dir = None
            self.activation_hook = None

    def _apply_init_perturbation(self, perturbation_seed: int, scale: float = 0.001) -> None:
        """
        对模型初始化应用微扰。
        
        :param perturbation_seed: 微扰的随机种子（与训练种子分离）
        :param scale: 微扰尺度，参数乘以 (1 + scale * noise)
        """
        print(f"[Perturbation] 应用初始化微扰: seed={perturbation_seed}, scale={scale}")
        
        # 使用独立的随机数生成器
        rng = torch.Generator()
        rng.manual_seed(perturbation_seed)
        
        total_params = 0
        perturbed_params = 0
        
        with torch.no_grad():
            for name, param in self.transformer.named_parameters():
                if param.requires_grad:
                    # 生成标准正态分布噪声
                    noise = torch.randn(param.shape, generator=rng, device=param.device, dtype=param.dtype)
                    # 应用乘性微扰: param = param * (1 + scale * noise)
                    param.mul_(1 + scale * noise)
                    perturbed_params += 1
                    total_params += param.numel()
        
        print(f"[Perturbation] 微扰了 {perturbed_params} 个参数张量，共 {total_params} 个参数值")

    def on_train_end(self) -> None:
        """训练结束时清理 hook"""
        if self.activation_hook is not None:
            self.activation_hook.remove()
            self.activation_hook = None

    def _activation_hook_fn(self, module, input, output):
        """Hook 函数：捕获最后一层 block 的输出（类方法，可序列化）"""
        # output 是 (a2, layer_attns, layer_values)
        # a2 的 shape 是 [batch_size, seq_len, d_model]
        if isinstance(output, tuple):
            activation = output[0]  # 取第一个元素（激活值）
        else:
            activation = output

        # 保存激活值（移动到 CPU 以便后续处理）
        self.captured_activation = activation.detach().cpu()

    def _register_activation_hook(self) -> None:
        """注册 forward hook 来捕获指定层的激活值"""
        if not self.enable_fingerprint:
            return

        # 获取最后一层 decoder block
        # decoder.blocks 是一个 ModuleList
        n_layers = self.hparams.n_layers  # type: ignore
        target_layer = self.transformer.decoder.blocks[n_layers - 1]
        # 注册 hook
        self.activation_hook = target_layer.register_forward_hook(self._activation_hook_fn)

    def _record_fingerprint_snapshot(self, step: int, epoch: int = None) -> None:
        """记录 fingerprint 快照"""
        if not self.enable_fingerprint:
            return

        import grok.fingerprint as fingerprint

        # 获取 probe 样本
        device = self.transformer.embedding.weight.device
        probe_indices_np = self.probe_indices.cpu().numpy()

        # 从训练数据集获取 probe 样本
        # 使用 numpy 索引，然后转换为 tensor
        probe_text = self.train_dataset.data[probe_indices_np, :-1].to(device)  # shape: [M, seq_len]
        
        # 获取正确答案的 token id（用于 log_prob 模式）
        # target 是从第二个 token 开始的序列，最后一个位置对应正确答案
        probe_target = self.train_dataset.data[probe_indices_np, 1:].to(device)  # shape: [M, seq_len]

        with torch.no_grad():
            if self.fingerprint_source == "log_prob":
                # log_prob 模式：获取每个样本正确答案位置的 log-prob
                logits, _, _ = self.transformer(probe_text, save_activations=False)
                # logits shape: [M, seq_len, vocab_len]
                
                # 计算 log_softmax（在 vocab 维度上）
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [M, seq_len, vocab_len]
                
                # 获取正确答案的 token id（最后一个位置）
                # probe_target[:, -1] 是每个样本最后一个 token 的正确答案
                correct_token_ids = probe_target[:, -1]  # shape: [M]
                
                # 提取每个样本在最后位置对正确答案的 log-prob
                # log_probs[:, -1, :] shape: [M, vocab_len]
                # 使用 gather 提取对应位置的值
                correct_log_probs = log_probs[:, -1, :].gather(
                    dim=1, index=correct_token_ids.unsqueeze(1)
                ).squeeze(1)  # shape: [M]
                
                # 转换为 numpy
                scalars_np = correct_log_probs.cpu().numpy()  # [M]
                
                # 计算分位数嵌入（一维分布）
                y = self.quantile_embedder.compute_embedding(scalars_np)  # [Q]
            else:
                # hidden_state 模式：使用 hook 捕获激活值
                self.captured_activation = None
                _ = self.transformer(probe_text, save_activations=False)

                if self.captured_activation is None:
                    print(f"警告：在 step {step} 未能捕获激活值")
                    return

                # 提取指定 token 位置的激活值
                # captured_activation shape: [M, seq_len, d_model]
                if self.token_index == -1:
                    activations = self.captured_activation[:, -1, :]  # [M, d_model]
                else:
                    activations = self.captured_activation[:, self.token_index, :]  # [M, d_model]

                # 转换为 numpy
                activations_np = activations.numpy()  # [M, d_model]

                # 计算分位数嵌入
                y = self.quantile_embedder.compute_embedding(activations_np)  # [J*Q]

        # 注意：中心化应该在分析时统一进行（跨时间步的均值），而不是训练时
        # 训练时保存原始快照，分析时会根据 center_mode 进行正确的中心化
        # 这里不再进行中心化，直接保存原始分位数嵌入向量
        # 如果需要在训练时中心化，应该在分析阶段统一处理所有快照
        
        # 保存快照（原始分位数嵌入向量，未中心化）
        fingerprint.save_snapshot(
            self.fingerprint_dir, y, step, epoch=epoch
        )

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        """
        Defines the hyperparameter arguments needed by instances of this
        class. This is intended to be called when parsing command line
        arguments.

        :param parser: an argparse.ArgumentParser created by the caller
        :returns: the argument parser with the command line arguments added
                  for this class.
        """
        parser.add_argument(
            "--batchsize",
            type=float,
            # default=0.25,
            default=0,
            help="-1 -> entire dataset, 0 -> auto-calculate, 0<N<1 -> fraction of dataset, N>1 -> N",
        )

        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--n_heads", type=int, default=4)
        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--weight_noise", type=float, default=0.0)
        parser.add_argument("--non_linearity", type=str, default="relu")
        parser.add_argument("--max_context_len", type=int, default=50)

        parser.add_argument("--math_operator", type=str, default="+")
        parser.add_argument(
            "--operand_length",
            type=int,
            help="for list operations, the length of the lists",
        )

        parser.add_argument("--train_data_pct", type=float, default=5)
        parser.add_argument("--warmup_steps", type=int, default=10)
        parser.add_argument("--anneal_lr_steps", type=int, default=100000)
        parser.add_argument("--anneal_lr", dest="anneal_lr", action="store_true")
        parser.set_defaults(anneal_lr=False)

        parser.add_argument("--max_lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--weight_decay_kind", type=str, default="to_zero")
        parser.add_argument("--noise_factor", type=float, default=0)

        parser.add_argument(
            "--save_activations", dest="save_activations", action="store_true"
        )
        parser.set_defaults(save_activations=False)
        parser.add_argument("--save_outputs", dest="save_outputs", action="store_true")
        parser.set_defaults(save_outputs=False)

        parser.add_argument(
            "--logdir",
            type=str,
            default=DEFAULT_LOG_DIR,
        )
        parser.add_argument(
            "--datadir",
            type=str,
            default=DEFAULT_DATA_DIR,
        )

        # Fingerprint 相关参数
        parser.add_argument(
            "--enable_fingerprint",
            dest="enable_fingerprint",
            action="store_true",
            help="启用分布转移算子谱指纹记录",
        )
        parser.set_defaults(enable_fingerprint=False)
        parser.add_argument(
            "--probe_size",
            type=int,
            default=100,
            help="Probe 样本数量",
        )
        parser.add_argument(
            "--proj_dim",
            type=int,
            default=100,
            help="随机投影方向数量 J",
        )
        parser.add_argument(
            "--n_quantiles",
            type=int,
            default=19,
            help="分位点数量 Q",
        )
        parser.add_argument(
            "--record_every",
            type=int,
            default=100,
            help="每 N 个训练 step 记录一次快照",
        )
        parser.add_argument(
            "--layer_name",
            type=str,
            default="decoder.blocks",
            help="要抽取激活值的层名称（相对于 transformer），默认最后一层 block",
        )
        parser.add_argument(
            "--token_index",
            type=int,
            default=-1,
            help="要抽取的 token 位置索引（-1 表示最后一个 token）",
        )
        parser.add_argument(
            "--center_mode",
            type=str,
            default="per_snapshot_mean",
            choices=["per_snapshot_mean", "relative_to_t0"],
            help="中心化模式：per_snapshot_mean（逐时刻减均值）或 relative_to_t0（相对 t=0）",
        )
        parser.add_argument(
            "--fingerprint_seed",
            type=int,
            default=42,
            help="Fingerprint 随机种子（用于固定投影方向）",
        )
        parser.add_argument(
            "--fingerprint_source",
            type=str,
            default="hidden_state",
            choices=["hidden_state", "log_prob"],
            help="Fingerprint 来源：hidden_state（隐状态）或 log_prob（log概率）",
        )
        parser.add_argument(
            "--early_stop_threshold",
            type=float,
            default=0,
            help="早停阈值（验证准确率，百分比，0表示禁用）",
        )
        parser.add_argument(
            "--early_stop_delay_steps",
            type=int,
            default=2000,
            help="达到阈值后继续训练的步数（默认：2000）",
        )
        parser.add_argument(
            "--perturbation_seed",
            type=int,
            default=None,
            help="初始化微扰的随机种子（None 表示不微扰）",
        )
        parser.add_argument(
            "--perturbation_scale",
            type=float,
            default=0.001,
            help="初始化微扰的尺度，参数乘以 (1 + scale * noise)，默认 0.001",
        )

        return parser

    def prepare_data(self) -> None:
        """
        Used by pytorch_lighting

        Loads training data to self.train_dataset
        Loads validation data to self.val_dataset
        """
        (self.train_dataset, self.val_dataset,) = ArithmeticDataset.splits(
            train_pct=self.hparams.train_data_pct,  # type: ignore
            operator=self.hparams.math_operator,  # type: ignore
            operand_length=self.hparams.operand_length,  # type: ignore
            data_dir=self.hparams.datadir,  # type: ignore
        )

    def train_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.train_dataset,
            device,
            batchsize_hint=self.hparams.batchsize,  # type: ignore
        )
        self.train_batchsize = iterator.batchsize
        self.batches_per_epoch = len(iterator)

        return iterator

    def val_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.val_dataset,
            device,
            batchsize_hint=-1,  # no need to batch validation data
        )
        return iterator

    def test_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.val_dataset, device, batchsize_hint=-1  # type: ignore
        )
        return iterator

    def _scheduler_lr(self, step: int) -> float:
        """
        Used by pytorch_lighting

        :returns: the learning_rate for this training step
        """
        max_lr = self.hparams.max_lr  # type: ignore
        min_lr = self.hparams.max_lr / 10  # type: ignore
        warmup_steps = self.hparams.warmup_steps  # type: ignore
        if not self.hparams.anneal_lr:
            if step <= warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            else:
                lr = max_lr
        else:
            if step <= warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            elif step <= self.hparams.anneal_lr_steps + warmup_steps:
                effective_step = step - warmup_steps
                t = effective_step / self.hparams.anneal_lr_steps
                cos = (1 + np.cos(np.pi * t)) / 2
                lr = min_lr + (max_lr - min_lr) * cos
                # lr = max_lr - ((effective_step / max_effective_step) * (max_lr - min_lr))
            else:
                lr = min_lr
        return lr

    def configure_optimizers(self) -> Tuple[List[Any], List[Dict]]:
        """
        Used by pytorch_lighting

        :returns: optimizers and schedulers.
        """
        optimizer = CustomAdamW(
            self.parameters(),
            betas=(0.9, 0.98),
            eps=1e-8,
            lr=1,
            weight_decay=self.hparams.weight_decay,
            noise_factor=self.hparams.noise_factor,
            weight_decay_form=self.hparams.weight_decay_kind,
        )
        # optimizer = SAM(
        #     self.parameters(),
        #     base_optimizer=CustomAdamW,
        #     rho=0.05,
        #     betas=(0.9, 0.98),
        #     eps=1e-8,
        #     lr=1,
        #     weight_decay=self.hparams.weight_decay,
        #     noise_factor=self.hparams.noise_factor,
        # )
        schedulers = [
            {
                "scheduler": LambdaLR(optimizer, lr_lambda=self._scheduler_lr),
                "interval": "step",
                "frequency": 1,
            }
        ]
        return [optimizer], schedulers

    def _accuracy(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Takes the most likely solution predicted for each equation and
        calculates the frac of equations in the batch for which these
        answers were correct

        :param y_hat: The softmax tensor output of the transformer
        :param y: A tensor of the token ids for the correct answers to each
                  equation in the batch
        :returns: the fraction of equations correctly answered
        """

        # find max prediction from output
        y_hat = torch.max(y_hat, dim=-2).indices  # batchsize x num_rhs_tokens
        # Strict whole-problem accuracy: all tokens must be correct
        row_accuracy = torch.min((y_hat == y), dim=-1).values  # shape: batchsize
        accuracy = row_accuracy.float() * 100  # shape: batchsize
        return accuracy
    
    def _token_accuracy(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Calculates token-average accuracy (per-token accuracy averaged across all tokens)

        :param y_hat: The softmax tensor output of the transformer
        :param y: A tensor of the token ids for the correct answers to each
                  equation in the batch
        :returns: token-average accuracy (scalar percentage)
        """
        # find max prediction from output
        y_hat = torch.max(y_hat, dim=-2).indices  # batchsize x num_rhs_tokens
        # Token-average accuracy: average over all tokens
        token_correct = (y_hat == y).float()  # batchsize x num_rhs_tokens
        token_accuracy = token_correct.mean() * 100  # scalar, percentage
        return token_accuracy

    def _step(
        self,
        batch: Dict,
        batch_idx: int,
        train: bool = True,
        reduction: str = "mean",
        grads: bool = False,
    ) -> Tuple[Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor]:
        """
        Performs one forward pass on a training or validation batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :param train: True is this is a training batch, false otherwise
        :returns: The loss from the predicted solutions to the equation,
                  The accuracy of the predicted solutions
                  The fraction of this dataset contained in this batch
                  The portion of the input equations left of the equal sign
                  The softmax probilities for the solutions to the equations
                  A list lists of attention matrices by layer and head
                  A list lists of value matrices by layer and head
                  Margin for this batch
        """
        x = batch["text"]  # shape = batchsize * context_len
        y = batch["target"]  # shape = batchsize * context_len
        y_hat, attentions, values = self(
            x=x, save_activations=self.hparams.save_activations  # type: ignore
        )  # shape = batchsize * context_len * vocab_size
        y_hat = y_hat.transpose(-2, -1)  # shape = batchsize * vocab_size * context_len

        # Note: each sample must have exactly one '=' and all of them must
        # have it in the same position.
        eq_token_index = self.train_dataset.tokenizer.stoi["="]
        eq_position_t = torch.nonzero(y[0, :] == eq_token_index, as_tuple=False)
        eq_position = int(eq_position_t.squeeze())

        # only calculate loss/accuracy on right hand side of the equation
        y_rhs = y[..., eq_position + 1 :]
        y_hat_rhs = y_hat[..., eq_position + 1 :]
        x_lhs = x[..., : eq_position + 1]

        if train:
            coeff = float(batch["target"].shape[0]) / len(self.train_dataset)
        else:
            coeff = float(batch["target"].shape[0]) / len(self.val_dataset)
        loss = F.cross_entropy(y_hat_rhs, y_rhs, reduction=reduction)

        with torch.no_grad():
            acc = self._accuracy(y_hat_rhs, y_rhs)
            token_acc = self._token_accuracy(y_hat_rhs, y_rhs)
            if reduction == "mean":
                acc = acc.mean()
            # token_acc is always a scalar, no need to handle reduction

        """
        device = self.transformer.embedding.weight.device
        self.margin = self.margin.to(device)

        output = y_hat_rhs.clone()  # batchsize, vocabsize, rhs tokens
        output_m = output.clone()  # batchsize, vocabsize, rhs tokens
        target = y_rhs.clone()  # batchsize, rhs tokens

        for i in range(output.size(0)):  # batch
            for j in range(output.size(2)):  # rhs tokens
                output_m[i, target[i, j], j] = output_m[i, :, j].min()

        for i in range(output.size(2)):  # rhs tokens
            output_compressed = output[:, target[:, i], i].squeeze().diag()
            output_m_compressed = (
                output_m[:, output_m.max(dim=1).indices[:, i], i].squeeze().diag()
            )
            self.margin = torch.cat(
                (
                    self.margin,
                    (output_compressed - output_m_compressed),
                ),
                0,
            )
        """
        grad_vec = None
        if grads:
            loss.backward()
            for p in self.parameters():
                p.grad.data.div_(batch["text"].shape[0])
                if grad_vec is None:
                    grad_vec = p.grad.data.view(-1)
                else:
                    grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))
            return loss, grad_vec
        return loss, acc, token_acc, coeff, x_lhs, y_hat_rhs, attentions, values


    def _save_inputs(self, outputs: Dict, ds: str) -> None:
        """
        Saves the input equations to disk for analysis later

        :param outputs: a list of tuples from self.training_step()
        :param ds: a string ('train' or 'val') naming which dataset
                   these inputs are from.
        :param train: True is this is a training batch, false otherwise
        """
        logdir = self.hparams.logdir + "/inputs/" + ds  # type: ignore
        os.makedirs(logdir, exist_ok=True)
        pickle_file = logdir + f"/{ds}.pt"

        x_lhs = torch.cat([x["x_lhs"] for x in outputs])
        with open(pickle_file, "wb") as fh:
            torch.save(x_lhs, fh)

    def _merge_batch_activations(
        self, partial_activations: List[List[Tensor]]
    ) -> List[List[Tensor]]:
        """
        Merges the head_attentions / head_values from all batches in
        this epoch.

        :param partial_activations: A list of
                                   (lists of lists of activations by layer and head)
        :returns: A lists of lists of activations by layer and head
        """
        # num_batches = len(partial_activations)
        num_layers = len(partial_activations[0])
        num_heads = len(partial_activations[0][0])
        activations: List = []
        for _ in range(num_layers):
            activations.append([])
            for _ in range(num_heads):
                activations[-1].append([])

        for minibatch_activations in partial_activations:
            for l, layer_activations in enumerate(minibatch_activations):
                for h, head_attn in enumerate(layer_activations):
                    # # print(f"head_attn = {head_attn}")
                    activations[l][h].append(head_attn)

        for l in range(num_layers):
            for h in range(num_heads):
                activations[l][h] = torch.cat(activations[l][h])

        return activations

    def _save_activations(self, outputs: Dict, ds: str) -> None:
        """
        Saves activations out to disk for analysis later

        :param outputs: a list of tuples from self.training_step()
        """

        output: Dict[str, Any] = {}
        if self.hparams.save_outputs:  # type: ignore
            y_hat_rhs = torch.cat([x["y_hat_rhs"] for x in outputs])
            output["y_hat_rhs"] = y_hat_rhs
        if self.hparams.save_activations:  # type: ignore
            partial_attentions = list([o["partial_attentions"] for o in outputs])
            attentions = self._merge_batch_activations(partial_attentions)
            partial_values = list([o["partial_values"] for o in outputs])
            values = self._merge_batch_activations(partial_values)
            output["attentions"] = attentions
            output["values"] = values
        if self.hparams.save_outputs or self.hparams.save_activations:  # type: ignore
            logdir = self.hparams.logdir + "/outputs/" + ds  # type: ignore
            os.makedirs(logdir, exist_ok=True)
            pickle_file = logdir + f"/epoch_{self.current_epoch:010}.pt"
            with open(pickle_file, "wb") as fh:
                torch.save(output, fh)

    def training_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward training pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with loss, accuracy, lr, probabilities of solutions,
                  attentions, and values
        """
        if batch_idx == 0:
            self.training_epoch_start_time = time.time()
            self.fwd_time_in_epoch = 0
            # Clear outputs from previous epoch
            self.training_step_outputs = []

        # 记录 fingerprint 快照（如果需要）
        if self.enable_fingerprint:
            global_step = self.global_step
            if global_step % self.record_every == 0:
                self._record_fingerprint_snapshot(global_step, epoch=self.current_epoch)

        start = time.time()
        loss, accuracy, token_accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
            batch=batch, batch_idx=batch_idx, train=True
        )
        self.fwd_time_in_epoch += time.time() - start

        # 每个epoch都收集数据，用于在epoch结束时记录
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        lr = optimizer.param_groups[0]["lr"]
        output = {
            "loss": loss,
            "partial_train_loss": coeff * loss,
            "partial_train_accuracy": coeff * accuracy,
            "partial_train_token_accuracy": token_accuracy,  # token_accuracy is already a scalar
            "learning_rate": torch.tensor([lr]),
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs

        # Store output for on_train_epoch_end (PyTorch Lightning 2.0+)
        # 每个epoch都收集，用于后续记录
        self.training_step_outputs.append(output)

        return output

    def on_train_epoch_end(self):
        """
        Used by pytorch_lightning
        Accumulates results of all forward training passes in this epoch
        """
        # Get outputs from training_step (PyTorch Lightning 2.0+)
        outputs = self.training_step_outputs
        
        # 每个epoch都记录训练指标，确保有足够的数据点绘制曲线
        epoch_is_to_be_logged = True  # 改为每个epoch都记录
        if epoch_is_to_be_logged and len(outputs) > 0:
            # 保持 next_train_epoch_to_log 更新，但不再用于控制是否记录
            self.next_train_epoch_to_log = self.current_epoch + 1
            with torch.no_grad():
                try:
                    loss = torch.stack([x["partial_train_loss"] for x in outputs]).sum()
                except Exception as e:
                    print("!" * 80)
                    print(outputs)
                    raise e
                perplexity = torch.exp(loss)
                accuracy = torch.stack(
                    [x["partial_train_accuracy"] for x in outputs]
                ).sum()
                # Token accuracy is already a scalar per batch, average across batches
                token_accuracy = torch.stack(
                    [x["partial_train_token_accuracy"] for x in outputs]
                ).mean()
            # avg_lr = torch.stack([x["learning_rate"] for x in outputs]).mean()
            # max_lr = torch.stack([x["learning_rate"] for x in outputs]).max()
            # last_lr = outputs[-1]["learning_rate"]
            first_lr = outputs[0]["learning_rate"]

            if self.hparams.save_activations or self.hparams.save_outputs:
                if self.current_epoch == 0:
                    self._save_inputs(outputs, ds="train")
                self._save_activations(outputs, ds="train")

            logs = {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_token_accuracy": token_accuracy,
                "train_perplexity": perplexity,
                "learning_rate": first_lr,
                "len_train_ds": len(self.train_dataset),
                "len_val_ds": len(self.val_dataset),
                "batches_per_epoch": self.batches_per_epoch,
                "time_per_epoch": time.time() - self.training_epoch_start_time,
                "fwd_time_in_epoch": self.fwd_time_in_epoch,
            }
            for k, v in logs.items():
                self.log(k, v)
        
        # Clear outputs for next epoch
        self.training_step_outputs = []

    def validation_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward validation pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy, probabilities of solutions,
                  attentions, and values
        """
        if self.next_epoch_to_eval < self.current_epoch:
            self.next_epoch_to_eval = self.current_epoch
            # Clear outputs for new epoch
            self.validation_step_outputs = []
        if self.current_epoch != self.next_epoch_to_eval:
            return {}
        with torch.no_grad():
            loss, accuracy, token_accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
                batch=batch, batch_idx=batch_idx, train=False
            )
        output = {
            "partial_val_loss": coeff * loss,
            "partial_val_accuracy": coeff * accuracy,
            "partial_val_token_accuracy": token_accuracy,  # token_accuracy is already a scalar
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs

        # Store output for on_validation_epoch_end (PyTorch Lightning 2.0+)
        self.validation_step_outputs.append(output)

        return output

    def on_validation_epoch_end(self):
        """
        Used by pytorch_lightning
        Accumulates results of all forward validation passes in this epoch
        """
        # Get outputs from validation_step (PyTorch Lightning 2.0+)
        outputs = self.validation_step_outputs
        validation_is_real = len(outputs) > 0 and len(outputs[0]) != 0

        if validation_is_real:
            self.next_epoch_to_eval = max(
                int(1.02 * self.next_epoch_to_eval), self.next_epoch_to_eval + 1
            )

            loss = torch.stack([x["partial_val_loss"] for x in outputs]).sum()
            perplexity = torch.exp(loss)
            accuracy = torch.stack([x["partial_val_accuracy"] for x in outputs]).sum()
            # Token accuracy is already a scalar per batch, average across batches
            token_accuracy = torch.stack([x["partial_val_token_accuracy"] for x in outputs]).mean()

            if self.hparams.save_activations or self.hparams.save_outputs:
                if self.current_epoch == 0:
                    self._save_inputs(outputs, ds="val")
                self._save_activations(outputs, ds="val")

            logs = {
                "val_loss": loss,
                "val_accuracy": accuracy,
                "val_token_accuracy": token_accuracy,
                "val_perplexity": perplexity,
            }
            
            # 保存当前验证准确率供CustomEarlyStopping Callback使用
            self.last_val_accuracy = accuracy.item()
            
            for name, param in self.named_parameters():
                # n parameters
                n_params = param.numel()
                # get the l2 norm of the parameter
                logs["paramnorm_" + name] = torch.norm(
                    param, 2
                ).detach().cpu().numpy() / np.sqrt(n_params)

            # train accuracy
            device = self.transformer.embedding.weight.device
            train_data = self.train_dataset.data.to(device)
            training_data = {"text": train_data[:, :-1], "target": train_data[:, 1:]}
            with torch.no_grad():
                tr_loss, tr_acc, tr_token_acc, *_ = self._step(training_data, 0)
                logs["full_train_loss"] = tr_loss
                logs["full_train_acc"] = tr_acc
                logs["full_train_token_acc"] = tr_token_acc

            for k, v in logs.items():
                self.log(k, v)
        
        # Clear outputs for next epoch
        self.validation_step_outputs = []
        
        # save a checkpoint if the epoch is a power of 2
        if (
            self.current_epoch > 0
            and int(2 ** (int(np.log(self.current_epoch) / np.log(2))))
            == self.current_epoch
        ):
            self.trainer.save_checkpoint(
                os.path.join(
                    self.hparams.checkpoint_path,
                    "epoch_" + str(self.current_epoch) + ".ckpt",
                ),
                weights_only=False,
            )
        if validation_is_real:
            return logs

    def test_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward validation pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy, probabilities of solutions,
                  attentions, and values
        """

        loss, accuracy, token_accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
            batch=batch, batch_idx=batch_idx, train=False, reduction="none"
        )
        # 注意：当 reduction="none" 时，loss 和 accuracy 都是逐样本的向量
        # token_accuracy 始终是标量（每个batch的平均）
        # 不需要乘以 coeff，直接保存向量，后续在 on_test_epoch_end 中聚合
        output = {
            "partial_test_loss": loss,  # 逐样本损失，shape: (batchsize,)
            "partial_test_accuracy": accuracy,  # 逐样本准确率，shape: (batchsize,)
            "partial_test_token_accuracy": token_accuracy,  # token准确率（标量）
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs

        # Store output for on_test_epoch_end (PyTorch Lightning 2.0+)
        self.test_step_outputs.append(output)

        return output

    def on_test_epoch_end(self):
        """
        Used by pytorch_lightning
        Accumulates results of all forward test passes in this epoch
        """
        # Get outputs from test_step (PyTorch Lightning 2.0+)
        outputs = self.test_step_outputs
        
        # 连接所有批次的逐样本损失和准确率
        loss_per_sample = torch.cat([x["partial_test_loss"] for x in outputs], dim=0)
        accuracy_per_sample = torch.cat([x["partial_test_accuracy"] for x in outputs], dim=0)
        
        # 聚合为标量：对损失和准确率取均值
        loss = loss_per_sample.mean()
        accuracy = accuracy_per_sample.mean()
        # Token accuracy is already a scalar per batch, average across batches
        token_accuracy = torch.stack([x["partial_test_token_accuracy"] for x in outputs]).mean()
        perplexity = torch.exp(loss)

        logs = {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_token_accuracy": token_accuracy,
            "test_perplexity": perplexity,
        }
        
        # Clear outputs
        self.test_step_outputs = []

        return {"test_loss": loss, "log": logs}

    def forward(self, *args, **kwargs) -> Any:
        """Passes all arguments directly to Tranformer.forward()"""
        return self.transformer(*args, **kwargs)


def train(hparams: Namespace) -> None:
    """
    This is the main trainer_method. This sets up and runs experiment with
    the defined hyperparameters

    :param hparams: An argparse.Namespace with all of the relevant hyperparameters
    """

    # Process the args
    if hparams.logdir is None:
        hparams.logdir = os.environ.get("LOGDIR", ".")
    hparams.logdir = os.path.abspath(hparams.logdir)

    # Make sure d_model, heads, and d_key are compatible
    assert (
        hparams.d_model % hparams.n_heads == 0
    ), "n_heads=%s does not evenly divide d_model=%s" % (
        hparams.n_heads,
        hparams.d_model,
    )
    hparams.d_key = hparams.d_model / hparams.n_heads

    # Set up the RNGs for repeatability
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    checkpoint_path = hparams.logdir + "/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    hparams.checkpoint_path = checkpoint_path

    # Create the model
    model = TrainableTransformer(hparams).float()

    torch.save(model, os.path.join(checkpoint_path, "init.pt"))

    logger = CSVLogger(hparams.logdir)

    # checkpointer = ModelCheckpoint(
    #     filepath=checkpoint_path,
    #     monitor="save_ckpt",
    #     mode="max",
    #     save_top_k=len(hparams.ckpt_epochs),
    #     verbose=False,
    # )

    # 添加自定义早停Callback
    callbacks_list = []
    early_stop_threshold = getattr(hparams, 'early_stop_threshold', 0)
    if early_stop_threshold > 0:
        early_stop_delay_steps = getattr(hparams, 'early_stop_delay_steps', 2000)
        callbacks_list.append(CustomEarlyStopping(threshold=early_stop_threshold, delay_steps=early_stop_delay_steps))

    trainer_args = {
        "max_steps": hparams.max_steps,
        "min_steps": hparams.max_steps,
        "max_epochs": int(1e8),
        "val_check_interval": 0.25,  # 降低验证频率：每25%的训练批次验证一次（相对值）
        "profiler": False,
        # "checkpoint_callback": checkpointer,
        "logger": logger,
        "log_every_n_steps": 1,
        "precision": "16-mixed",  # 启用混合精度训练（FP16）
        "callbacks": callbacks_list,  # 添加早停Callback
        # "flush_logs_every_n_steps" removed in PyTorch Lightning 2.6+
    }
    if torch.cuda.is_available() and hparams.gpu >= 0:
        trainer_args["accelerator"] = "gpu"
        trainer_args["devices"] = [hparams.gpu]
    else:
        trainer_args["accelerator"] = "cpu"

    trainer = Trainer(**trainer_args)

    trainer.fit(model=model)  # type: ignore
    """
    margin = np.percentile(model.margin.detach().cpu().numpy(), 5)
    device = transformer.embedding.weight.device
    measures, bounds = metrics.calculate(
        transformer,
        transformer_init.to(device),
        device,
        dataset_size,
        margin,
        input_dim=hparams.d_model,
    )

    measures_file = os.path.join(logger.log_dir, "measures.json")
    bounds_file = os.path.join(logger.log_dir, "bounds.json")
    with open(measures_file, "w") as fh:
        json.dump(measures, fh)
    with open(bounds_file, "w") as fh:
        json.dump(bounds, fh)
    """
    return hparams.logdir


def compute_sharpness(hparams: Namespace, ckpts) -> None:
    """
    This is the compute_sharpness method. This loads a series of checkpoints in
    the defined hyperparameters

    :param hparams: An argparse.Namespace with all of the relevant hyperparameters
    """

    # Process the args
    if hparams.logdir is None:
        hparams.logdir = os.environ.get("LOGDIR", ".")
    hparams.logdir = os.path.abspath(hparams.logdir)

    # Make sure d_model, heads, and d_key are compatible
    assert (
        hparams.d_model % hparams.n_heads == 0
    ), "n_heads=%s does not evenly divide d_model=%s" % (
        hparams.n_heads,
        hparams.d_model,
    )
    hparams.d_key = hparams.d_model / hparams.n_heads

    # Set up the RNGs for repeatability
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    checkpoint_path = hparams.logdir + "/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    hparams.checkpoint_path = checkpoint_path

    # Create the model
    model = TrainableTransformer(hparams).float()

    torch.save(model, os.path.join(checkpoint_path, "init.pt"))

    logger = CSVLogger(hparams.logdir)


    trainer_args = {
        "max_steps": hparams.max_steps,
        "min_steps": hparams.max_steps,
        "max_epochs": int(1e8),
        "val_check_interval": 1,
        "profiler": False,
        # "checkpoint_callback": checkpointer,
        "logger": logger,
        "log_every_n_steps": 1,
        # "flush_logs_every_n_steps" removed in PyTorch Lightning 2.6+
    }
    if torch.cuda.is_available() and hparams.gpu >= 0:
        trainer_args["accelerator"] = "gpu"
        trainer_args["devices"] = [hparams.gpu]
    else:
        trainer_args["accelerator"] = "cpu"

    trainer = Trainer(**trainer_args)

    for ckpt in ckpts:
        print(f"Loading checkpoint {ckpt}")
        # model = torch.load(ckpt)
        # model.load_state_dict(torch.load(ckpt))

        checkpoint = torch.load(ckpt)
        # print(dir(checkpoint), type(checkpoint), "Ckpt")
        # for k, v in checkpoint.items():
        #     print(k)
        # print(checkpoint["hyper_parameters"])

        hps = checkpoint["hyper_parameters"]
        hps = argparse.Namespace(**hps)
        model = TrainableTransformer(hps).float()
        model.load_state_dict(checkpoint["state_dict"])

        phi = get_sharpness(model.train_dataloader(), model)
        results = {}
        results[ckpt] = phi
        pickle.dump(results, open(f"results/results_SD-{i}.pkl", "wb"))


def add_args(parser=None) -> Namespace:
    """
    Parses the command line arguments

    :returns: an argparse.Namespace with all of the needed arguments
    """
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=100000)
    # parser.add_argument("--checkpoint_period", type=int, default=1)
    parser = TrainableTransformer.add_model_specific_args(parser)
    return parser


class CustomAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        noise_factor=0.0,
        weight_decay_form="to_zero",
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not weight_decay_form in ["to_zero", "to_init", "jiggle", "honest"]:
            raise ValueError(
                f"Invalid weight decay form: {weight_decay_form}, should be one of ['to_zero', 'to_init', 'jiggle']"
            )
        # if not 0.0 <= weight_decay:
        #     raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            noise_factor=noise_factor,
            weight_decay_form=weight_decay_form,
        )
        super(CustomAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CustomAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad

                if group["weight_decay"] > 0:
                    if group["weight_decay_form"] == "honest":
                        grad = grad + group["weight_decay"] * p.detach()

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["weight_decay_form"] == "to_init":
                        state["init"] = p.detach().clone()
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                if group["weight_decay"] > 0:
                    if group["weight_decay_form"] == "to_zero":
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    elif group["weight_decay_form"] == "to_init":
                        p.add_(
                            (state["init"] - p) * (group["lr"] * group["weight_decay"])
                        )
                    elif group["weight_decay_form"] == "jiggle":
                        p.mul_(
                            torch.exp(
                                torch.randn(1).cuda()
                                * (group["lr"] * group["weight_decay"])
                            )
                        )
                    elif group["weight_decay_form"] == "honest":
                        pass
                    else:
                        raise ValueError(
                            f"Invalid weight decay form: {group['weight_decay_form']}"
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )

                step_size = group["lr"] / bias_correction1

                upd = exp_avg / denom
                # add uniform gaussian noise to the update
                if group["noise_factor"] > 0:
                    upd += torch.randn_like(upd) * group["noise_factor"]
                # if group['noise_factor'] > 0:
                #     upd *= torch.exp(torch.randn_like(upd) * group['noise_factor'])
                p.add_(-step_size * upd)

        return loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        grad_norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        print("grad norms is ", grad_norms, "!" * 1000)
        norm = torch.norm(
            torch.stack(grad_norms),
            p=2,
        )
        return norm
