"""
工具函数：保存/加载、probe 选择等
"""

import os
import json
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


def select_probe_samples(
    model_or_dataset,
    probe_size: int,
    random_seed: int = 42,
    split: str = "train",
) -> torch.Tensor:
    """
    从数据集中选择固定的 probe 样本。

    :param model_or_dataset: 模型对象（有 train_dataset 属性）或数据集对象
    :param probe_size: 选择的样本数量
    :param random_seed: 随机种子
    :param split: 从哪个集合选择 ("train" 或 "val")
    :returns: 选择的样本索引 tensor
    """
    np.random.seed(random_seed)

    # 如果是模型对象，获取数据集
    if hasattr(model_or_dataset, "train_dataset"):
        if split == "train":
            data_source = model_or_dataset.train_dataset
        else:
            data_source = model_or_dataset.val_dataset
    else:
        # 直接是数据集对象
        data_source = model_or_dataset

    total_size = len(data_source)
    probe_size = min(probe_size, total_size)

    # 随机选择固定索引
    indices = np.random.choice(total_size, size=probe_size, replace=False)
    indices = np.sort(indices)  # 排序以保证可复现

    return torch.tensor(indices, dtype=torch.long)


def save_fingerprint_config(
    output_dir: str,
    config: Dict[str, Any],
) -> None:
    """
    保存 fingerprint 配置。

    :param output_dir: 输出目录
    :param config: 配置字典
    """
    os.makedirs(output_dir, exist_ok=True)
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_fingerprint_config(output_dir: str) -> Dict[str, Any]:
    """
    加载 fingerprint 配置。

    :param output_dir: 输入目录
    :returns: 配置字典
    """
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_probe_indices(output_dir: str, probe_indices: np.ndarray) -> None:
    """
    保存 probe 样本索引。

    :param output_dir: 输出目录
    :param probe_indices: probe 索引数组
    """
    os.makedirs(output_dir, exist_ok=True)
    indices_file = os.path.join(output_dir, "probe_indices.npy")
    np.save(indices_file, probe_indices)


def load_probe_indices(output_dir: str) -> np.ndarray:
    """
    加载 probe 样本索引。

    :param output_dir: 输入目录
    :returns: probe 索引数组
    """
    indices_file = os.path.join(output_dir, "probe_indices.npy")
    return np.load(indices_file)


def save_projection_directions(output_dir: str, directions: np.ndarray) -> None:
    """
    保存投影方向。

    :param output_dir: 输出目录
    :param directions: 投影方向矩阵，shape [J, d_model]
    """
    os.makedirs(output_dir, exist_ok=True)
    proj_file = os.path.join(output_dir, "proj_directions.npy")
    np.save(proj_file, directions)


def load_projection_directions(output_dir: str) -> np.ndarray:
    """
    加载投影方向。

    :param output_dir: 输入目录
    :returns: 投影方向矩阵，shape [J, d_model]
    """
    proj_file = os.path.join(output_dir, "proj_directions.npy")
    return np.load(proj_file)


def save_snapshot(
    output_dir: str,
    snapshot: np.ndarray,
    step: int,
    epoch: int = None,
) -> None:
    """
    保存快照（中心化后的分位数嵌入向量）。

    :param output_dir: 输出目录
    :param snapshot: 快照向量，shape [J*Q]
    :param step: 训练步数
    :param epoch: 训练轮数（可选）
    """
    os.makedirs(output_dir, exist_ok=True)
    snapshot_file = os.path.join(output_dir, f"snapshot_t{step:06d}.npy")
    np.save(snapshot_file, snapshot)

    # 更新 steps.npy
    steps_file = os.path.join(output_dir, "steps.npy")
    epochs_file = os.path.join(output_dir, "epochs.npy")
    if os.path.exists(steps_file):
        try:
            steps = np.load(steps_file).tolist()
            epochs = np.load(epochs_file).tolist() if os.path.exists(epochs_file) else []
        except (EOFError, ValueError):
            # 文件损坏或为空，重新初始化
            steps = []
            epochs = []
    else:
        steps = []
        epochs = []

    if step not in steps:
        steps.append(step)
        epochs.append(epoch if epoch is not None else -1)
        # 按 step 排序
        sorted_idx = np.argsort(steps)
        steps = [steps[i] for i in sorted_idx]
        epochs = [epochs[i] for i in sorted_idx]
        np.save(steps_file, np.array(steps))
        np.save(os.path.join(output_dir, "epochs.npy"), np.array(epochs))


def load_snapshots(output_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载所有快照。

    :param output_dir: 输入目录
    :returns: (snapshots, steps, epochs)
              snapshots: shape [T, J*Q]
              steps: shape [T]
              epochs: shape [T]
    """
    steps_file = os.path.join(output_dir, "steps.npy")
    if not os.path.exists(steps_file):
        raise FileNotFoundError(f"未找到 steps.npy 文件: {steps_file}")

    steps = np.load(steps_file)
    epochs_file = os.path.join(output_dir, "epochs.npy")
    if os.path.exists(epochs_file):
        epochs = np.load(epochs_file)
    else:
        epochs = np.full(len(steps), -1)

    snapshots = []
    for step in steps:
        snapshot_file = os.path.join(output_dir, f"snapshot_t{int(step):06d}.npy")
        if not os.path.exists(snapshot_file):
            raise FileNotFoundError(f"未找到快照文件: {snapshot_file}")
        snapshot = np.load(snapshot_file)
        snapshots.append(snapshot)

    snapshots = np.array(snapshots)  # shape: [T, J*Q]

    return snapshots, steps, epochs