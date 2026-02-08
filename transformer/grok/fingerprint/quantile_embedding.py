"""
分位数嵌入模块：实现随机投影 + 1D 分位数的 sliced 表征
以及标量分布的分位数嵌入（用于 log-prob 模式）
"""

import numpy as np
import torch
from typing import Tuple


class ScalarQuantileEmbedding:
    """
    将一维标量分布转换为分位数嵌入向量。
    用于 log-prob 模式：每个样本的正确答案 log-prob 构成一维分布。
    """

    def __init__(
        self,
        n_quantiles: int = 19,
        quantile_range: Tuple[float, float] = (0.05, 0.95),
    ):
        """
        初始化标量分位数嵌入器。

        :param n_quantiles: 分位点数量 Q
        :param quantile_range: 分位点范围 (q_min, q_max)，默认 (0.05, 0.95)
        """
        self.n_quantiles = n_quantiles
        self.quantile_range = quantile_range

        # 生成分位点
        q_min, q_max = quantile_range
        self.quantiles = np.linspace(q_min, q_max, n_quantiles)  # shape: [Q]

    def compute_embedding(self, scalars: np.ndarray) -> np.ndarray:
        """
        计算分位数嵌入向量。

        :param scalars: 标量数组，shape [M]，M 是样本数量
        :returns: 分位数嵌入向量，shape [Q]
        """
        assert scalars.ndim == 1, f"期望一维数组，实际维度: {scalars.ndim}"
        
        # 计算经验分位点
        q_values = np.quantile(scalars, self.quantiles)  # shape: [Q]
        return q_values

    def center_embedding(
        self, y: np.ndarray, reference: np.ndarray = None
    ) -> np.ndarray:
        """
        中心化嵌入向量。

        :param y: 嵌入向量，shape [Q]
        :param reference: 参考向量（用于相对中心化），如果为 None 则使用自身均值
        :returns: 中心化后的向量
        """
        if reference is None:
            # 逐时刻减去自身均值
            y_centered = y - np.mean(y)
        else:
            # 以参考向量为基点
            y_centered = y - reference
        return y_centered

    def get_output_dim(self) -> int:
        """返回输出维度 Q"""
        return self.n_quantiles


class QuantileEmbedding:
    """
    将高维激活值转换为分位数嵌入向量。
    使用随机投影 + 1D 分位数的 sliced 表征方法。
    """

    def __init__(
        self,
        d_model: int,
        n_projections: int = 100,
        n_quantiles: int = 19,
        quantile_range: Tuple[float, float] = (0.05, 0.95),
        random_seed: int = 42,
    ):
        """
        初始化分位数嵌入器。

        :param d_model: 激活值的维度
        :param n_projections: 随机投影方向的数量 J
        :param n_quantiles: 分位点数量 Q
        :param quantile_range: 分位点范围 (q_min, q_max)，默认 (0.05, 0.95)
        :param random_seed: 随机种子，用于固定投影方向
        """
        self.d_model = d_model
        self.n_projections = n_projections
        self.n_quantiles = n_quantiles
        self.quantile_range = quantile_range
        self.random_seed = random_seed

        # 生成固定的随机投影方向
        np.random.seed(random_seed)
        # 生成 J 个随机单位向量
        projections = np.random.randn(n_projections, d_model)
        # 归一化为单位向量
        norms = np.linalg.norm(projections, axis=1, keepdims=True)
        self.projection_directions = projections / norms  # shape: [J, d_model]

        # 生成分位点
        q_min, q_max = quantile_range
        self.quantiles = np.linspace(q_min, q_max, n_quantiles)  # shape: [Q]

    def compute_embedding(self, activations: np.ndarray) -> np.ndarray:
        """
        计算分位数嵌入向量。

        :param activations: 激活值数组，shape [M, d_model]，M 是样本数量
        :returns: 分位数嵌入向量 y，shape [J*Q]
        """
        M, d_model = activations.shape
        assert d_model == self.d_model, f"维度不匹配: 期望 {self.d_model}, 实际 {d_model}"

        # 对每个投影方向计算标量样本
        # activations: [M, d_model]
        # projection_directions: [J, d_model]
        # scores: [J, M]
        scores = activations @ self.projection_directions.T  # [M, d_model] @ [d_model, J] = [M, J]
        scores = scores.T  # [J, M]

        # 对每个投影方向的标量样本计算分位点
        quantile_features = []
        for j in range(self.n_projections):
            sample_values = scores[j, :]  # shape: [M]
            # 计算经验分位点
            q_values = np.quantile(sample_values, self.quantiles)  # shape: [Q]
            quantile_features.append(q_values)

        # 拼接所有 (j, q) 得到向量 y
        y = np.concatenate(quantile_features, axis=0)  # shape: [J*Q]

        return y

    def compute_embedding_torch(self, activations: torch.Tensor) -> np.ndarray:
        """
        计算分位数嵌入向量（PyTorch 版本）。

        :param activations: 激活值张量，shape [M, d_model]
        :returns: 分位数嵌入向量 y，shape [J*Q]
        """
        # 转换为 numpy
        if activations.is_cuda:
            activations = activations.cpu()
        activations_np = activations.detach().numpy()
        return self.compute_embedding(activations_np)

    def center_embedding(
        self, y: np.ndarray, reference: np.ndarray = None
    ) -> np.ndarray:
        """
        中心化嵌入向量（投影到切空间）。

        :param y: 嵌入向量，shape [J*Q]
        :param reference: 参考向量（用于相对中心化），如果为 None 则使用自身均值
        :returns: 中心化后的向量
        """
        if reference is None:
            # 逐时刻减去自身均值
            y_centered = y - np.mean(y)
        else:
            # 以参考向量为基点
            y_centered = y - reference
        return y_centered

    def get_output_dim(self) -> int:
        """返回输出维度 J*Q"""
        return self.n_projections * self.n_quantiles