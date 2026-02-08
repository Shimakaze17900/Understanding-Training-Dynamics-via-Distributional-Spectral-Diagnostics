#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析分布式快照实验结果并可视化
训练完成后自动运行
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from dynamic_mode_decomposition import time_delay, data_matrices, dmd

# ==================== 全局参数 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
exp_path = os.path.join(current_dir, 'Results', '')
save_path = os.path.join(current_dir, 'Figures', '')
datapath = os.path.join(exp_path, 'SGD_MNIST_')

# 实验参数
h_values = [5, 40, 256]
random_seeds = np.array([2])
M = 1  # 初始化数量
T = 1000  # 1个epoch的时间步数（每个batch记录一次，约1000个batch）

# DMD参数
n_delays = 32
k_mode = 10

# 分位点设置（19个分位点）
n_quantiles = 19
quantiles = np.linspace(0.05, 0.95, n_quantiles)

# 绘图参数
seed_plot = 0
save_flag = True

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

# ==================== 加载数据并处理 ====================

print("加载权重数据（h=5, 40, 256）...")
all_X_dist = {h: [] for h in h_values}
valid_seeds = []

for ii in range(len(random_seeds)):
    print(f"处理种子 {ii+1}/{len(random_seeds)} (seed={random_seeds[ii]})...")
    
    weights_all_inits = {h: [] for h in h_values}
    
    for h in h_values:
        for jj in range(M):
            try:
                W_path = datapath + f'h={h}_seed' + str(random_seeds[ii]) + '_initialization' + str(jj) + '_weights.npy'
                
                if not os.path.exists(W_path):
                    print(f"  跳过: {W_path} 不存在")
                    continue
                
                W = np.load(W_path)
                
                # 确保时间步数正确
                if W.shape[0] > T:
                    W = W[:T, :]
                elif W.shape[0] < T:
                    print(f"  警告: h={h} 只有 {W.shape[0]} 个时间步，少于预期的 {T}")
                
                weights_all_inits[h].append(W)
                
            except Exception as e:
                print(f"  错误: {e}")
                continue
    
    # 检查是否有足够的数据
    if all(len(weights_all_inits[h]) > 0 for h in h_values):
        # 对所有初始化取平均
        for h in h_values:
            W_avg = np.mean(np.array(weights_all_inits[h]), axis=0)
            
            # 构造经验分布快照
            snapshots = construct_distribution_snapshots(W_avg, quantiles)
            
            # 跨step中心化
            snapshots_centered = cross_step_centering(snapshots)
            
            all_X_dist[h].append(snapshots_centered)
        
        valid_seeds.append(ii)

print(f"\n成功加载 {len(valid_seeds)} 个种子的数据")

if len(valid_seeds) == 0:
    print("错误: 没有找到任何权重数据！")
    exit(1)

# ==================== DMD分析 ====================

print("\n进行DMD分析...")
all_eigs = {h: [] for h in h_values}

for ii in range(len(valid_seeds)):
    print(f"DMD分析种子 {ii+1}/{len(valid_seeds)} (原始种子索引={valid_seeds[ii]})...")
    
    for h in h_values:
        if len(all_X_dist[h]) <= ii:
            continue
        
        X_dist = all_X_dist[h][ii]
        T_actual, D_quantiles = X_dist.shape
        
        # 准备数据
        X_reshaped = X_dist[:, :, np.newaxis]
        
        # 时间延迟嵌入
        n_delays_final = min(n_delays, T_actual // 2)
        if T_actual > n_delays_final:
            Z_final = time_delay(X_reshaped, n_delays_final)
            Z_final, Z_prime_final = data_matrices(Z_final)
            eigs_final, _ = dmd(Z_final.T, Z_prime_final.T, k=k_mode)
            all_eigs[h].append(eigs_final)
        else:
            all_eigs[h].append(np.array([]))

# 过滤空的特征值数组
all_eigs_valid = {h: [e for e in all_eigs[h] if len(e) > 0] for h in h_values}

print(f"\n成功完成DMD分析:")
for h in h_values:
    print(f"  h={h}: {len(all_eigs_valid[h])}个种子")

# ==================== 绘制谱点图 ====================

if all(len(all_eigs_valid[h]) > seed_plot for h in h_values):
    print(f"\n绘制谱点图（使用种子索引 {seed_plot}）...")
    
    unit_circle_x = np.sin(np.arange(0, 2 * np.pi, 0.01))
    unit_circle_y = np.cos(np.arange(0, 2 * np.pi, 0.01))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(unit_circle_x, unit_circle_y, 'k--', label='Unit circle', linewidth=1)
    
    colors = {5: 'k', 40: 'g', 256: 'r'}
    markers = {5: 'o', 40: 's', 256: '^'}
    labels = {5: 'h=5', 40: 'h=40', 256: 'h=256'}
    
    for h in h_values:
        if len(all_eigs_valid[h]) > seed_plot:
            eigs = all_eigs_valid[h][seed_plot]
            plt.plot(np.real(eigs), np.imag(eigs), 
                    color=colors[h], marker=markers[h], 
                    label=labels[h], markersize=8, alpha=0.7, linestyle='None')
    
    plt.xlabel(r'Real($\lambda$)', fontsize=12)
    plt.ylabel(r'Imag($\lambda$)', fontsize=12)
    plt.xlim([0.90, 1.05])  # 放大范围
    plt.ylim([-0.10, 0.10])  # 放大范围
    plt.legend(fontsize=11)
    plt.title('Koopman Eigenvalues (Distribution Snapshots: h=5,40,256)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    
    if save_flag:
        fig.savefig(os.path.join(save_path, 'Koopman_eigenvalues_distribution_h=5_40_256.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"谱点图已保存")
    
    plt.show()
    
    # 保存特征值
    if save_flag:
        os.makedirs(exp_path, exist_ok=True)
        for h in h_values:
            np.save(os.path.join(exp_path, f'Koopman_eigenvalues_distribution_h={h}.npy'), all_eigs_valid[h])
        print("特征值已保存")

print("\n可视化分析完成！")
