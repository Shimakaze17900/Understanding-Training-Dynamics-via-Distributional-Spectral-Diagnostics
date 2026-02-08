"""
Hankel-DMD-RRR 模块：实现 time-delay embedding + DMD with reduced rank regression
"""

import numpy as np
from typing import Tuple, Optional


def time_delay(X: np.ndarray, n_delays: int) -> np.ndarray:
    """
    对时间序列进行 time-delay embedding (Hankel 嵌入)。

    :param X: 输入时间序列，shape [T, d]，T 是时间步数，d 是特征维度
    :param n_delays: time-delay 参数（嵌入窗口大小）
    :returns: Hankel 矩阵，shape [T - n_delays + 1, n_delays * d]
    """
    T, d = X.shape
    if T < n_delays:
        raise ValueError(f"时间步数 T={T} 必须 >= n_delays={n_delays}")

    # 构建 Hankel 矩阵
    # 每一行是 [x(t), x(t+1), ..., x(t+n_delays-1)] 的拼接
    X_td = []
    for t in range(T - n_delays + 1):
        window = X[t : t + n_delays, :].flatten()  # shape: [n_delays * d]
        X_td.append(window)

    X_td = np.array(X_td)  # shape: [T - n_delays + 1, n_delays * d]
    return X_td


def data_matrices(X_td: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 time-delay 嵌入构建数据矩阵 Z 和 Z'（用于 DMD）。

    :param X_td: time-delay 嵌入矩阵，shape [T_delayed, n_delays * d]
    :returns: (Z, Zp)，其中 Z 是 [t, t+1, ..., t+T_delayed-2]，Zp 是 [t+1, t+2, ..., t+T_delayed-1]
    """
    T_delayed, d_embed = X_td.shape
    if T_delayed < 2:
        raise ValueError("time-delay 序列长度必须 >= 2")

    # Z: 从 0 到 T_delayed-2
    Z = X_td[:-1, :].T  # shape: [d_embed, T_delayed - 1]

    # Z': 从 1 到 T_delayed-1
    Zp = X_td[1:, :].T  # shape: [d_embed, T_delayed - 1]

    return Z, Zp


def dmd_rrr(
    Z: np.ndarray, Zp: np.ndarray, k_mode: int = 10, eps: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    执行 DMD with Reduced Rank Regression (RRR)。

    :param Z: 数据矩阵，shape [d_embed, T-1]
    :param Zp: 数据矩阵（延迟一步），shape [d_embed, T-1]
    :param k_mode: SVD 截断 rank
    :param eps: 奇异值阈值，小于此值的奇异值将被丢弃
    :returns: (eigenvalues, eigenvectors, modes)
              eigenvalues: shape [k_mode]（复数）
              eigenvectors: shape [d_embed, k_mode]（右特征向量）
              modes: shape [k_mode, d_embed]（左特征向量）
    """
    d_embed, T_minus_1 = Z.shape
    assert Zp.shape == (d_embed, T_minus_1), "Z 和 Zp 的维度必须匹配"

    # 计算 Z 的 SVD
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    # U: [d_embed, min(d_embed, T_minus_1)]
    # S: [min(d_embed, T_minus_1)]
    # Vt: [min(d_embed, T_minus_1), T_minus_1]

    # 截断到 k_mode
    k_actual = min(k_mode, len(S))
    # 过滤掉太小的奇异值
    # 使用相对阈值：如果最大奇异值很小，则使用更宽松的阈值
    max_sv = S[0] if len(S) > 0 else 0
    if max_sv < 1e-8:
        # 如果最大奇异值很小，使用更宽松的相对阈值
        relative_eps = max(eps, max_sv * 1e-6)
        valid_idx = S > relative_eps
    else:
        valid_idx = S > eps
    k_actual = min(k_actual, np.sum(valid_idx))
    if k_actual == 0:
        raise ValueError(f"所有奇异值都小于阈值（最大奇异值: {max_sv:.2e}），无法进行 DMD。"
                         f"这可能是因为数据变化太小或数据全为0。")

    U_k = U[:, :k_actual]  # [d_embed, k_actual]
    S_k = S[:k_actual]  # [k_actual]
    Vt_k = Vt[:k_actual, :]  # [k_actual, T_minus_1]

    # 计算 A_tilde = U^T Z' V Σ^{-1}
    # 更标准的做法是: A_tilde = U^T @ Zp @ V @ diag(1/S)
    V_k = Vt_k.T  # [T_minus_1, k_actual]
    S_inv = 1.0 / S_k  # [k_actual]

    # A_tilde = U^T @ Zp @ V @ diag(S_inv)
    # 形状: [k_actual, d_embed] @ [d_embed, T_minus_1] @ [T_minus_1, k_actual] @ diag([k_actual])
    # = [k_actual, k_actual]
    A_tilde = U_k.T @ Zp @ V_k @ np.diag(S_inv)

    # 计算 A_tilde 的特征值和特征向量
    eigenvals, eigenvecs_right = np.linalg.eig(A_tilde)
    # eigenvals: [k_actual]（复数）
    # eigenvecs_right: [k_actual, k_actual]（列向量是特征向量）

    # 从右特征向量恢复原始空间的模态
    # modes = U @ eigenvecs_right
    modes = U_k @ eigenvecs_right  # [d_embed, k_actual]

    return eigenvals, modes, eigenvecs_right


def apply_pca(
    X: np.ndarray, pca_dim: Optional[int] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    对时间序列应用 PCA 降维（可选，使用 numpy 实现）。

    :param X: 输入时间序列，shape [T, d]
    :param pca_dim: PCA 目标维度，如果为 None 则不应用 PCA
    :returns: (X_pca, pca_components)
              X_pca: shape [T, pca_dim] 或 [T, d]（如果 pca_dim 为 None）
              pca_components: shape [d, pca_dim]（如果应用了 PCA）或 None
    """
    if pca_dim is None:
        return X, None

    # 使用 numpy 实现简单的 PCA
    # 中心化
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - X_mean

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # U: [T, min(T, d)]
    # S: [min(T, d)]
    # Vt: [min(T, d), d]

    # 取前 pca_dim 个主成分
    k = min(pca_dim, Vt.shape[0])
    V_k = Vt[:k, :].T  # [d, k]，列向量是主成分

    # 投影到主成分空间
    X_pca = X_centered @ V_k  # [T, d] @ [d, k] = [T, k]

    return X_pca, V_k  # 返回主成分矩阵（列向量是主成分）