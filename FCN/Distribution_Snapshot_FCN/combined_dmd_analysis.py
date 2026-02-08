#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨种子堆叠的总DMD分析
将所有种子的快照堆叠起来，进行总的DMD分析
然后计算每250个时间步的重构误差和有效秩
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from dynamic_mode_decomposition import time_delay, data_matrices, dmd

# ==================== 全局参数 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
exp_path = os.path.join(current_dir, 'Results', '')
save_path = os.path.join(current_dir, 'combined_analysis', '')
datapath = os.path.join(exp_path, 'SGD_MNIST_')

# 实验参数
h_values = [5, 40, 256, 1024]
random_seeds = np.array([2, 3, 6, 7, 11, 16, 17, 22, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101])  # 25个种子
M = 1  # 每个种子1个初始化
T = 1000  # 1个epoch的时间步数
record_interval = 250  # 每250个时间步记录一次
n_records = T // record_interval  # 记录次数：4次（250, 500, 750, 1000）

# DMD参数
n_delays = 32
k_mode = 10

# 分位点设置（19个分位点）
n_quantiles = 19
quantiles = np.linspace(0.05, 0.95, n_quantiles)

# 有效秩阈值
fixed_threshold = 1e-6
energy_threshold = 0.99  # 99%能量

save_flag = True

# 创建输出文件夹
os.makedirs(save_path, exist_ok=True)

# ==================== 辅助函数 ====================

def construct_distribution_snapshots(weights, quantiles):
    """从原始权重构造经验分布快照"""
    T, D = weights.shape
    n_quantiles = len(quantiles)
    snapshots = np.zeros((T, n_quantiles))
    
    for t in range(T):
        snapshots[t, :] = np.quantile(weights[t, :], quantiles)
    
    return snapshots

def cross_step_centering(snapshots):
    """跨step中心化"""
    means = np.mean(snapshots, axis=0, keepdims=True)
    centered = snapshots - means
    return centered

def compute_dmd_reconstruction_error(Z, Z_prime, eigs, modes, k):
    """
    计算DMD重构误差
    
    参数:
        Z: 输入数据矩阵 (n_features, n_samples)
        Z_prime: 目标数据矩阵 (n_features, n_samples)
        eigs: DMD特征值
        modes: DMD模式
        k: 使用的模式数量
    
    返回:
        reconstruction_error: 重构误差（归一化Frobenius范数）
    """
    if k > len(eigs):
        k = len(eigs)
    
    modes_k = modes[:, :k]
    eigs_k = eigs[:k]
    
    # 伪逆
    modes_pinv = np.linalg.pinv(modes_k)
    
    # 重构
    Z_recon = np.zeros_like(Z_prime, dtype=complex)
    for i in range(Z.shape[1]):
        z_t = Z[:, i]
        coeff_t = modes_pinv @ z_t.reshape(-1, 1)
        z_t1_pred = modes_k @ (eigs_k.reshape(-1, 1) * coeff_t)
        Z_recon[:, i] = z_t1_pred.flatten()
    
    # 转换为实数
    Z_recon = np.real(Z_recon)
    
    # 计算重构误差（归一化Frobenius范数）
    reconstruction_error = np.linalg.norm(Z_prime - Z_recon, 'fro') / (np.linalg.norm(Z_prime, 'fro') + 1e-10)
    
    return reconstruction_error

def compute_effective_rank(S, fixed_threshold=1e-6, energy_threshold=0.99):
    """
    计算有效秩
    
    参数:
        S: 奇异值数组（已按降序排列）
        fixed_threshold: 固定阈值（默认1e-6），应用于原始奇异值的绝对值
        energy_threshold: 能量阈值（默认0.99，即99%）
    
    返回:
        rank_fixed: 基于固定阈值的有效秩（原始奇异值 > fixed_threshold）
        rank_energy: 基于能量阈值的有效秩（累积能量达到总能量99%）
        S_max: 最大奇异值
        S_min: 最小奇异值
    """
    if len(S) == 0:
        return 0, 0, 0, 0
    
    S_max = S[0] if len(S) > 0 else 0
    S_min = S[-1] if len(S) > 0 else 0
    
    # 方法1：固定阈值（原始奇异值的绝对值）
    rank_fixed = np.sum(S > fixed_threshold)
    
    # 方法2：99%能量阈值
    # 计算总能量（使用原始奇异值的平方）
    total_energy = np.sum(S ** 2)
    if total_energy > 0:
        cumulative_energy = np.cumsum(S ** 2)
        # 找到累积能量达到总能量99%的位置
        rank_energy = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
        # 确保不超过实际秩
        rank_energy = min(rank_energy, len(S))
    else:
        rank_energy = 0
    
    return rank_fixed, rank_energy, S_max, S_min

# ==================== 加载数据并堆叠 ====================

print("加载权重数据并堆叠所有种子的快照...")
all_combined_snapshots = {h: None for h in h_values}

for h in h_values:
    print(f"\n处理 h={h}...")
    all_snapshots_list = []
    
    for seed_idx, seed in enumerate(random_seeds):
        for jj in range(M):
            try:
                W_path = datapath + f'h={h}_seed' + str(seed) + '_initialization' + str(jj) + '_weights.npy'
                
                if not os.path.exists(W_path):
                    print(f"  跳过: {W_path} 不存在")
                    continue
                
                W = np.load(W_path)
                
                if W.shape[0] > T:
                    W = W[:T, :]
                elif W.shape[0] < T:
                    print(f"  警告: seed={seed} 只有 {W.shape[0]} 个时间步")
                
                # 构造经验分布快照
                snapshots = construct_distribution_snapshots(W, quantiles)
                
                # 跨step中心化
                snapshots_centered = cross_step_centering(snapshots)
                
                all_snapshots_list.append(snapshots_centered)
                print(f"  种子 {seed}: 快照形状 {snapshots_centered.shape}")
                
            except Exception as e:
                print(f"  错误 (seed={seed}): {e}")
                continue
    
    if len(all_snapshots_list) > 0:
        # 堆叠所有种子的快照（沿时间维度堆叠）
        # 形状: (n_seeds * T, n_quantiles)
        combined_snapshots = np.vstack(all_snapshots_list)
        all_combined_snapshots[h] = combined_snapshots
        print(f"  堆叠后形状: {combined_snapshots.shape} (共 {len(all_snapshots_list)} 个种子)")
    else:
        print(f"  警告: h={h} 没有找到任何数据")

print(f"\n成功堆叠所有种子的快照")

# ==================== 计算每250步的DMD重构误差和有效秩 ====================

print(f"\n计算每{record_interval}个时间步的DMD重构误差和有效秩（跨种子堆叠数据）...")

all_reconstruction_errors = {h: [] for h in h_values}
all_effective_ranks_fixed = {h: [] for h in h_values}
all_effective_ranks_energy = {h: [] for h in h_values}
all_S_max = {h: [] for h in h_values}
all_S_min = {h: [] for h in h_values}

for h in h_values:
    if all_combined_snapshots[h] is None:
        continue
    
    X_combined = all_combined_snapshots[h]
    T_combined, D_quantiles = X_combined.shape
    
    print(f"\n处理 h={h} (总时间步数: {T_combined})...")
    
    reconstruction_errors = []
    effective_ranks_fixed = []
    effective_ranks_energy = []
    S_max_values = []
    S_min_values = []
    
    # 在每个记录点计算（相对于原始时间步，不是堆叠后的时间步）
    # 但我们需要在堆叠后的数据上计算，所以记录点应该基于堆叠后的时间步
    # 每个记录点对应原始时间步的250, 500, 750, 1000
    # 在堆叠数据中，这些对应 250, 500, 750, 1000, 1250, 1500, ... (每个种子1000步)
    
    # 实际上，我们应该在每个原始时间步的250, 500, 750, 1000处计算
    # 但由于是堆叠的，我们需要考虑所有种子的对应位置
    
    # 方案：在每个记录点，使用从开始到该点的所有数据
    for record_idx in range(n_records):
        # 记录点对应的时间步范围（在原始时间步中）
        start_t_original = record_idx * record_interval
        end_t_original = min((record_idx + 1) * record_interval, T)
        
        # 在堆叠数据中，我们需要包含所有种子在这个时间范围内的数据
        # 堆叠后的数据形状是 (n_seeds * T, n_quantiles)
        # 所以我们需要提取每个种子的 [start_t_original:end_t_original] 部分，然后堆叠
        
        n_seeds_loaded = T_combined // T  # 实际加载的种子数
        
        # 提取每个种子在对应时间范围内的数据
        window_snapshots = []
        for seed_idx in range(n_seeds_loaded):
            seed_start = seed_idx * T
            seed_end = seed_idx * T + T
            seed_data = X_combined[seed_start:seed_end, :]
            window_data = seed_data[start_t_original:end_t_original, :]
            window_snapshots.append(window_data)
        
        # 堆叠所有种子在这个时间窗口的数据
        X_window = np.vstack(window_snapshots)
        
        if X_window.shape[0] < n_delays + 10:
            reconstruction_errors.append(np.nan)
            effective_ranks_fixed.append(np.nan)
            effective_ranks_energy.append(np.nan)
            S_max_values.append(np.nan)
            S_min_values.append(np.nan)
            continue
        
        # 准备数据
        X_reshaped = X_window[:, :, np.newaxis]
        
        # 时间延迟嵌入
        n_delays_window = min(n_delays, X_window.shape[0] // 2)
        if n_delays_window < 2:
            reconstruction_errors.append(np.nan)
            effective_ranks_fixed.append(np.nan)
            effective_ranks_energy.append(np.nan)
            S_max_values.append(np.nan)
            S_min_values.append(np.nan)
            continue
        
        Z = time_delay(X_reshaped, n_delays_window)
        Z, Z_prime = data_matrices(Z)
        
        # DMD分析
        try:
            # 先进行SVD以获取奇异值
            U, S, Vh = np.linalg.svd(Z.T, full_matrices=False)
            
            # 计算有效秩和奇异值范围
            rank_fixed, rank_energy, S_max, S_min = compute_effective_rank(S, fixed_threshold, energy_threshold)
            effective_ranks_fixed.append(rank_fixed)
            effective_ranks_energy.append(rank_energy)
            S_max_values.append(S_max)
            S_min_values.append(S_min)
            
            # 使用前k_mode个模式进行DMD
            k_use = min(k_mode, len(S))
            eigs, modes = dmd(Z.T, Z_prime.T, k=k_use)
            
            # 计算重构误差
            recon_error = compute_dmd_reconstruction_error(Z.T, Z_prime.T, eigs, modes, k_use)
            reconstruction_errors.append(recon_error)
            
            print(f"  记录点{record_idx+1} (时间步{start_t_original}-{end_t_original}): "
                  f"重构误差={recon_error:.6f}, "
                  f"有效秩(固定)={rank_fixed}, 有效秩(99%能量)={rank_energy}, "
                  f"最大奇异值={S_max:.4e}, 最小奇异值={S_min:.4e}")
            
        except Exception as e:
            print(f"    记录点{record_idx+1} 计算失败: {e}")
            reconstruction_errors.append(np.nan)
            effective_ranks_fixed.append(np.nan)
            effective_ranks_energy.append(np.nan)
            S_max_values.append(np.nan)
            S_min_values.append(np.nan)
    
    # 保存结果
    all_reconstruction_errors[h] = reconstruction_errors
    all_effective_ranks_fixed[h] = effective_ranks_fixed
    all_effective_ranks_energy[h] = effective_ranks_energy
    all_S_max[h] = S_max_values
    all_S_min[h] = S_min_values

# ==================== 进行总的DMD分析（用于谱点图） ====================

print("\n进行总的DMD分析（跨种子堆叠，用于谱点图）...")
all_eigs_combined = {}

for h in h_values:
    if all_combined_snapshots[h] is None:
        continue
    
    X_combined = all_combined_snapshots[h]
    T_combined, D_quantiles = X_combined.shape
    
    print(f"处理 h={h} (总时间步数: {T_combined})...")
    
    # 准备数据
    X_reshaped = X_combined[:, :, np.newaxis]
    
    # 时间延迟嵌入
    n_delays_final = min(n_delays, T_combined // 2)
    if T_combined > n_delays_final:
        Z_final = time_delay(X_reshaped, n_delays_final)
        Z_final, Z_prime_final = data_matrices(Z_final)
        eigs_final, _ = dmd(Z_final.T, Z_prime_final.T, k=k_mode)
        all_eigs_combined[h] = eigs_final
        print(f"  成功: 提取了 {len(eigs_final)} 个特征值")
    else:
        all_eigs_combined[h] = np.array([])
        print(f"  失败: 数据不足")

# ==================== 绘制谱点图 ====================

print("\n绘制谱点图（跨种子堆叠）...")

unit_circle_x = np.sin(np.arange(0, 2 * np.pi, 0.01))
unit_circle_y = np.cos(np.arange(0, 2 * np.pi, 0.01))

fig, ax = plt.subplots(figsize=(10, 8))
plt.plot(unit_circle_x, unit_circle_y, 'k--', label='Unit circle', linewidth=1)

colors = {5: 'k', 40: 'g', 256: 'r', 1024: 'b'}
markers = {5: 'o', 40: 's', 256: '^', 1024: 'd'}
labels = {5: 'h=5', 40: 'h=40', 256: 'h=256', 1024: 'h=1024'}

for h in h_values:
    if h in all_eigs_combined and len(all_eigs_combined[h]) > 0:
        eigs = all_eigs_combined[h]
        plt.plot(np.real(eigs), np.imag(eigs), 
                color=colors[h], marker=markers[h], 
                label=labels[h], 
                markersize=8, alpha=0.7, linestyle='None')

plt.xlabel(r'Real($\lambda$)', fontsize=12)
plt.ylabel(r'Imag($\lambda$)', fontsize=12)
plt.xlim([0.80, 1.20])
plt.ylim([-0.25, 0.25])
plt.legend(fontsize=11)
plt.title('Koopman Eigenvalues (Combined Across Seeds: h=5,40,256,1024)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

if save_flag:
    fig.savefig(os.path.join(save_path, 'Koopman_eigenvalues_combined.png'), 
               dpi=300, bbox_inches='tight')
    print("谱点图已保存")

plt.close()

# ==================== 输出结果到Markdown文档 ====================

print("\n生成Markdown报告...")

md_content = f"""# 跨种子堆叠的总DMD分析报告

## 实验设置
- 优化器: SGD (lr=0.1)
- 网络宽度: h = 5, 40, 256, 1024
- 随机种子: {', '.join(map(str, random_seeds))} (共{len(random_seeds)}个)
- 训练epochs: 1
- 总训练样本: 60,000
- 每个种子时间步数: {T}
- **堆叠后总时间步数**: {T * len(random_seeds)} (所有种子堆叠)
- 记录间隔: 每{record_interval}个时间步（基于原始时间步）
- 分位点数量: {n_quantiles}
- DMD模式数: {k_mode}
- 固定阈值: {fixed_threshold}
- 能量阈值: {energy_threshold * 100}%

## 数据堆叠说明
所有种子的快照沿时间维度堆叠，形成更大的数据集进行DMD分析。
每个记录点的计算使用所有种子在该时间范围内的堆叠数据。

## DMD重构误差和有效秩

### 记录点说明
每{record_interval}个时间步记录一次（基于原始时间步）。记录点对应的时间步：
"""

# 添加记录点时间步
for i in range(n_records):
    start_step = i * record_interval
    end_step = min((i + 1) * record_interval, T)
    samples_start = start_step * 60
    samples_end = end_step * 60
    md_content += f"- 记录点 {i+1}: 时间步 {start_step} - {end_step} (约 {samples_start} - {samples_end} 个样本)\n"

md_content += "\n### 详细结果\n\n"

# 为每个网络宽度创建表格
for h in h_values:
    if h not in all_reconstruction_errors or len(all_reconstruction_errors[h]) == 0:
        continue
    
    md_content += f"#### h = {h}\n\n"
    md_content += "| 记录点 | 时间步范围 | 重构误差 | 有效秩(固定阈值) | 有效秩(99%能量) | 最大奇异值 | 最小奇异值 |\n"
    md_content += "|--------|-----------|---------|-----------------|----------------|-----------|-----------|\n"
    
    errors = all_reconstruction_errors[h]
    ranks_fixed = all_effective_ranks_fixed[h]
    ranks_energy = all_effective_ranks_energy[h]
    S_max_vals = all_S_max[h]
    S_min_vals = all_S_min[h]
    
    for i in range(len(errors)):
        start_step = i * record_interval
        end_step = min((i + 1) * record_interval, T)
        
        error_val = errors[i]
        rank_fixed_val = ranks_fixed[i]
        rank_energy_val = ranks_energy[i]
        S_max_val = S_max_vals[i]
        S_min_val = S_min_vals[i]
        
        if not np.isnan(error_val):
            md_content += f"| {i+1} | {start_step}-{end_step} | {error_val:.6f} | {int(rank_fixed_val)} | {int(rank_energy_val)} | {S_max_val:.4e} | {S_min_val:.4e} |\n"
        else:
            md_content += f"| {i+1} | {start_step}-{end_step} | N/A | N/A | N/A | N/A | N/A |\n"
    
    md_content += "\n"

md_content += "### 统计摘要\n\n"

# 计算统计信息
for h in h_values:
    if h not in all_reconstruction_errors or len(all_reconstruction_errors[h]) == 0:
        continue
    
    md_content += f"#### h = {h}\n\n"
    
    errors = all_reconstruction_errors[h]
    ranks_fixed = all_effective_ranks_fixed[h]
    ranks_energy = all_effective_ranks_energy[h]
    S_max_vals = all_S_max[h]
    S_min_vals = all_S_min[h]
    
    valid_errors = [e for e in errors if not np.isnan(e)]
    valid_ranks_fixed = [r for r in ranks_fixed if not np.isnan(r)]
    valid_ranks_energy = [r for r in ranks_energy if not np.isnan(r)]
    valid_S_max = [s for s in S_max_vals if not np.isnan(s)]
    valid_S_min = [s for s in S_min_vals if not np.isnan(s)]
    
    if len(valid_errors) > 0:
        md_content += f"**重构误差统计：**\n"
        md_content += f"- 平均重构误差: {np.mean(valid_errors):.6f}\n"
        md_content += f"- 标准差: {np.std(valid_errors):.6f}\n"
        md_content += f"- 最小重构误差: {np.min(valid_errors):.6f}\n"
        md_content += f"- 最大重构误差: {np.max(valid_errors):.6f}\n\n"
        
        md_content += f"**有效秩统计（固定阈值{fixed_threshold}）：**\n"
        md_content += f"- 平均有效秩: {np.mean(valid_ranks_fixed):.2f}\n"
        md_content += f"- 标准差: {np.std(valid_ranks_fixed):.2f}\n"
        md_content += f"- 最小有效秩: {int(np.min(valid_ranks_fixed))}\n"
        md_content += f"- 最大有效秩: {int(np.max(valid_ranks_fixed))}\n\n"
        
        md_content += f"**有效秩统计（99%能量阈值）：**\n"
        md_content += f"- 平均有效秩: {np.mean(valid_ranks_energy):.2f}\n"
        md_content += f"- 标准差: {np.std(valid_ranks_energy):.2f}\n"
        md_content += f"- 最小有效秩: {int(np.min(valid_ranks_energy))}\n"
        md_content += f"- 最大有效秩: {int(np.max(valid_ranks_energy))}\n\n"
        
        md_content += f"**奇异值范围统计：**\n"
        md_content += f"- 最大奇异值 - 平均: {np.mean(valid_S_max):.4e}, 范围: [{np.min(valid_S_max):.4e}, {np.max(valid_S_max):.4e}]\n"
        md_content += f"- 最小奇异值 - 平均: {np.mean(valid_S_min):.4e}, 范围: [{np.min(valid_S_min):.4e}, {np.max(valid_S_min):.4e}]\n\n"

md_content += "\n## 说明\n\n"
md_content += "- **数据堆叠**：所有种子的快照沿时间维度堆叠，形成更大的数据集\n"
md_content += "- **重构误差**：使用Frobenius范数归一化的重构误差\n"
md_content += f"- **有效秩（固定阈值）**：原始奇异值大于{fixed_threshold}的数量\n"
md_content += f"- **有效秩（99%能量）**：累积能量达到总能量{energy_threshold*100}%所需的奇异值数量\n"
md_content += "- **最大/最小奇异值**：SVD分解得到的最大和最小奇异值\n"
md_content += "- N/A表示该记录点数据不足，无法进行DMD分析\n"

# 保存Markdown文件
md_file = os.path.join(save_path, 'DMD_combined_analysis.md')
with open(md_file, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"Markdown报告已保存: {md_file}")

# 保存数值结果
if save_flag:
    os.makedirs(save_path, exist_ok=True)
    for h in h_values:
        if h in all_reconstruction_errors:
            np.save(os.path.join(save_path, f'DMD_reconstruction_errors_combined_h={h}.npy'), all_reconstruction_errors[h])
            np.save(os.path.join(save_path, f'DMD_effective_ranks_fixed_combined_h={h}.npy'), all_effective_ranks_fixed[h])
            np.save(os.path.join(save_path, f'DMD_effective_ranks_energy_combined_h={h}.npy'), all_effective_ranks_energy[h])
            np.save(os.path.join(save_path, f'DMD_S_max_combined_h={h}.npy'), all_S_max[h])
            np.save(os.path.join(save_path, f'DMD_S_min_combined_h={h}.npy'), all_S_min[h])
        if h in all_eigs_combined:
            np.save(os.path.join(save_path, f'Koopman_eigenvalues_combined_h={h}.npy'), all_eigs_combined[h])
    print("数值结果已保存")

print("\n跨种子堆叠的总DMD分析完成！")
