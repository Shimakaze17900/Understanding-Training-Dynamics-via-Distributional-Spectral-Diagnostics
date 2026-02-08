#!/usr/bin/env python
"""
分析 fingerprint 快照序列并生成谱图。

用法:
    python scripts/analyze_fingerprint.py --run_dir <logdir> [options]
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from typing import Optional

import grok.fingerprint as fingerprint


def plot_spectrum(
    eigenvalues: np.ndarray,
    output_path: str,
    title: str = "DMD-RRR 特征值谱",
) -> None:
    """
    绘制特征值谱图（复平面散点图 + 单位圆）。

    :param eigenvalues: 特征值数组，shape [k_mode]（复数）
    :param output_path: 输出文件路径
    :param title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # 提取实部和虚部
    real_parts = eigenvalues.real
    imag_parts = eigenvalues.imag

    # 绘制特征值散点
    ax.scatter(real_parts, imag_parts, s=50, alpha=0.7, c='blue', edgecolors='black', linewidths=0.5)
    ax.set_xlabel("实部 (Re)", fontsize=12)
    ax.set_ylabel("虚部 (Im)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 绘制单位圆
    theta = np.linspace(0, 2 * np.pi, 1000)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)
    ax.plot(unit_circle_x, unit_circle_y, 'r--', linewidth=2, label='单位圆', alpha=0.7)

    # 设置坐标轴比例相等
    ax.set_aspect('equal', adjustable='box')

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加图例
    ax.legend(loc='upper right')

    # 添加坐标轴
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

    # 设置坐标轴范围（稍微超出单位圆）
    max_range = max(1.2, np.max(np.abs(real_parts)), np.max(np.abs(imag_parts))) * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"谱图已保存至: {output_path}")

    # 同时保存 PDF 版本
    if output_path.endswith('.png'):
        pdf_path = output_path.replace('.png', '.pdf')
    else:
        pdf_path = output_path + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF 版本已保存至: {pdf_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="分析 fingerprint 快照序列并生成 DMD-RRR 谱图"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="运行目录（包含 fingerprints/ 子目录）",
    )
    parser.add_argument(
        "--n_delays",
        type=int,
        default=32,
        help="time-delay 参数（默认: 32）",
    )
    parser.add_argument(
        "--k_mode",
        type=int,
        default=10,
        help="DMD-RRR SVD 截断 rank（默认: 10）",
    )
    parser.add_argument(
        "--center_mode",
        type=str,
        default="per_snapshot_mean",
        choices=["per_snapshot_mean", "relative_to_t0"],
        help="中心化模式（默认: per_snapshot_mean）",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="PCA 降维维度（可选，默认: None，不降维）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认: run_dir/fingerprints/）",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=None,
        help="最大步数（过滤掉超过此步数的快照，默认: None，不过滤）",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=None,
        help="最小步数（过滤掉小于此步数的快照，默认: None，不过滤）",
    )
    parser.add_argument(
        "--segment_size",
        type=int,
        default=None,
        help="分段大小（每N步绘制一张谱图，默认: None，不分段）",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="输出文件名（默认: spectrum.png，分段时会添加步数范围后缀）",
    )
    parser.add_argument(
        "--normalize_scale",
        action="store_true",
        help="去尺度处理：对每个快照进行单位化（除以L2范数），消除绝对尺度，只保留形状信息",
    )

    args = parser.parse_args()

    # 确定输入和输出目录
    fingerprint_dir = os.path.join(args.run_dir, "fingerprints")
    if not os.path.exists(fingerprint_dir):
        raise FileNotFoundError(f"未找到 fingerprints 目录: {fingerprint_dir}")

    output_dir = args.output_dir if args.output_dir else fingerprint_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"加载快照序列从: {fingerprint_dir}")

    # 加载快照序列
    snapshots, steps, epochs = fingerprint.load_snapshots(fingerprint_dir)
    print(f"已加载 {len(snapshots)} 个快照（原始）")
    print(f"步数范围（原始）: {steps.min()} - {steps.max()}")
    
    # 如果指定了min_step或max_step，过滤快照
    if args.min_step is not None or args.max_step is not None:
        mask = np.ones(len(steps), dtype=bool)
        if args.min_step is not None:
            mask = mask & (steps >= args.min_step)
        if args.max_step is not None:
            mask = mask & (steps <= args.max_step)
        snapshots = snapshots[mask]
        steps = steps[mask]
        epochs = epochs[mask] if epochs is not None else None
        print(f"过滤后: {len(snapshots)} 个快照")
        if len(snapshots) > 0:
            print(f"步数范围（过滤后）: {steps.min()} - {steps.max()}")
    
    # 检查数据是否全为0
    if len(snapshots) > 0:
        data_std = np.std(snapshots)
        if data_std < 1e-10:
            print(f"\n警告: 快照数据标准差极小 ({data_std:.2e})，可能所有值都接近0")
            print("这通常发生在模型完全收敛后，所有log-prob值变得相同")
            print("无法进行DMD分析，将跳过此段")
            return

    # 加载配置
    try:
        config = fingerprint.load_fingerprint_config(fingerprint_dir)
        J = config.get('n_projections', 100)
        Q = config.get('n_quantiles', 19)
        fingerprint_source = config.get('fingerprint_source', 'hidden_state')
    except:
        # 如果无法加载配置，尝试从快照维度推断
        J = 100  # 默认值
        Q = snapshots.shape[1] // J if snapshots.shape[1] > 100 else snapshots.shape[1]
        fingerprint_source = 'hidden_state'
        print(f"警告：无法加载配置，使用默认值 J={J}, Q={Q}")
    
    print(f"指纹模式: {fingerprint_source}, J={J}, Q={Q}")
    
    # 去尺度处理（如果需要）
    if args.normalize_scale:
        if fingerprint_source == "log_prob":
            # log_prob 模式：快照是 Q 维，直接对整个向量归一化
            print("应用去尺度处理（对整个分位点向量归一化）...")
            norms = np.linalg.norm(snapshots, axis=1, keepdims=True)  # [T, 1]
            norms = np.where(norms > 1e-10, norms, 1.0)
            snapshots = snapshots / norms
            print(f"去尺度后分位点向量范数范围: [{np.linalg.norm(snapshots, axis=1).min():.6f}, {np.linalg.norm(snapshots, axis=1).max():.6f}]")
        else:
            # hidden_state 模式：对每个投影方向的分位点向量归一化
            print("应用去尺度处理（对每个投影方向的分位点向量归一化）...")
            # 将快照重塑为 [T, J, Q]
            snapshots_reshaped = snapshots.reshape(len(snapshots), J, Q)
            # 对每个投影方向的分位点向量进行归一化
            norms = np.linalg.norm(snapshots_reshaped, axis=2, keepdims=True)  # [T, J, 1]
            norms = np.where(norms > 1e-10, norms, 1.0)
            snapshots_reshaped = snapshots_reshaped / norms  # [T, J, Q]
            # 重塑回 [T, J*Q]
            snapshots = snapshots_reshaped.reshape(len(snapshots), J * Q)
            print(f"去尺度后每个投影方向分位点向量范数范围: [{np.linalg.norm(snapshots_reshaped, axis=2).min():.6f}, {np.linalg.norm(snapshots_reshaped, axis=2).max():.6f}]")

    # 检查是否有足够的快照进行 time-delay embedding
    if len(snapshots) < args.n_delays + 1:
        raise ValueError(
            f"快照数量 ({len(snapshots)}) 不足以进行 time-delay embedding "
            f"(需要至少 {args.n_delays + 1} 个快照)"
        )

    # 中心化处理：对应分位点跨时间步的均值
    # 这是正确的中心化方式：每个分位点位置减去该位置在所有时间步上的均值
    print("应用中心化处理（对应分位点跨时间步的均值）...")
    snapshots = snapshots - np.mean(snapshots, axis=0, keepdims=True)
    print(f"中心化后数据范围: [{snapshots.min():.6f}, {snapshots.max():.6f}]")
    print(f"中心化后数据标准差: {snapshots.std():.6f}")

    # 分段处理
    if args.segment_size is not None:
        # 分段绘制谱图
        max_step = steps.max()
        segment_start = 0
        segment_idx = 0
        
        while segment_start < max_step:
            segment_end = min(segment_start + args.segment_size, max_step)
            
            # 过滤当前段的快照
            segment_mask = (steps >= segment_start) & (steps <= segment_end)
            segment_snapshots = snapshots[segment_mask]
            segment_steps = steps[segment_mask]
            
            if len(segment_snapshots) < args.n_delays + 1:
                print(f"\n跳过段 {segment_start}-{segment_end}: 快照数量不足 ({len(segment_snapshots)} < {args.n_delays + 1})")
                segment_start = segment_end
                continue
            
            print(f"\n处理段 {segment_start}-{segment_end}: {len(segment_snapshots)} 个快照")
            
            # 中心化处理：对应分位点跨时间步的均值（对当前段）
            print("应用中心化处理（对应分位点跨时间步的均值）...")
            segment_snapshots = segment_snapshots - np.mean(segment_snapshots, axis=0, keepdims=True)
            
            # 去尺度处理（如果需要，在分段时也需要应用）
            if args.normalize_scale:
                if fingerprint_source == "log_prob":
                    # log_prob 模式：快照是 Q 维，直接对整个向量归一化
                    norms = np.linalg.norm(segment_snapshots, axis=1, keepdims=True)
                    norms = np.where(norms > 1e-10, norms, 1.0)
                    segment_snapshots = segment_snapshots / norms
                else:
                    # hidden_state 模式：对每个投影方向的分位点向量归一化
                    # 将快照重塑为 [T_seg, J, Q]
                    segment_snapshots_reshaped = segment_snapshots.reshape(len(segment_snapshots), J, Q)
                    # 对每个投影方向的分位点向量进行归一化
                    norms = np.linalg.norm(segment_snapshots_reshaped, axis=2, keepdims=True)  # [T_seg, J, 1]
                    norms = np.where(norms > 1e-10, norms, 1.0)
                    segment_snapshots_reshaped = segment_snapshots_reshaped / norms
                    # 重塑回 [T_seg, J*Q]
                    segment_snapshots = segment_snapshots_reshaped.reshape(len(segment_snapshots), J * Q)
            
            # 应用 PCA（如果需要）
            if args.pca_dim is not None:
                print(f"应用 PCA 降维到 {args.pca_dim} 维...")
                from grok.fingerprint.dmd_rrr import apply_pca
                segment_snapshots, pca_components = apply_pca(segment_snapshots, args.pca_dim)
            
            # Time-delay embedding
            print(f"应用 time-delay embedding (n_delays={args.n_delays})...")
            from grok.fingerprint.dmd_rrr import time_delay, data_matrices, dmd_rrr
            X_td = time_delay(segment_snapshots, args.n_delays)
            
            # 构建数据矩阵
            print("构建数据矩阵...")
            Z, Zp = data_matrices(X_td)
            
            # DMD-RRR
            print(f"执行 DMD-RRR (k_mode={args.k_mode})...")
            eigenvalues, modes, eigenvecs_right = dmd_rrr(Z, Zp, k_mode=args.k_mode)
            print(f"计算得到 {len(eigenvalues)} 个特征值")
            
            # 保存特征值
            base_filename = args.output_filename if args.output_filename else "spectrum"
            if base_filename.endswith('.png'):
                base_filename = base_filename[:-4]
            eigenvalues_path = os.path.join(output_dir, f"{base_filename}_step{segment_start:05d}-{segment_end:05d}_eigenvalues.npy")
            np.save(eigenvalues_path, eigenvalues)
            
            # 打印特征值统计信息
            print(f"\n特征值统计信息 (段 {segment_start}-{segment_end}):")
            print(f"  数量: {len(eigenvalues)}")
            print(f"  模长范围: [{np.abs(eigenvalues).min():.4f}, {np.abs(eigenvalues).max():.4f}]")
            print(f"  实部范围: [{eigenvalues.real.min():.4f}, {eigenvalues.real.max():.4f}]")
            print(f"  虚部范围: [{eigenvalues.imag.min():.4f}, {eigenvalues.imag.max():.4f}]")
            print(f"  单位圆内特征值数量: {np.sum(np.abs(eigenvalues) < 1.0)}")
            print(f"  单位圆上特征值数量: {np.sum(np.abs(np.abs(eigenvalues) - 1.0) < 0.01)}")
            print(f"  单位圆外特征值数量: {np.sum(np.abs(eigenvalues) > 1.0)}")
            
            # 绘制谱图
            spectrum_path = os.path.join(output_dir, f"{base_filename}_step{segment_start:05d}-{segment_end:05d}.png")
            scale_suffix = " (去尺度)" if args.normalize_scale else ""
            title = f"DMD-RRR Spectrum (steps {segment_start}-{segment_end}, n_delays={args.n_delays}, k_mode={args.k_mode}){scale_suffix}"
            plot_spectrum(eigenvalues, spectrum_path, title=title)
            
            segment_start = segment_end
            segment_idx += 1
        
        print(f"\n分段分析完成！共生成 {segment_idx} 个谱图")
    else:
        # 不分段，处理全部数据
        # 应用 PCA（如果需要）
        if args.pca_dim is not None:
            print(f"应用 PCA 降维到 {args.pca_dim} 维...")
            from grok.fingerprint.dmd_rrr import apply_pca
            snapshots, pca_components = apply_pca(snapshots, args.pca_dim)
            print(f"降维后形状: {snapshots.shape}")

        # Time-delay embedding
        print(f"应用 time-delay embedding (n_delays={args.n_delays})...")
        from grok.fingerprint.dmd_rrr import time_delay, data_matrices, dmd_rrr
        X_td = time_delay(snapshots, args.n_delays)
        print(f"Time-delay 嵌入形状: {X_td.shape}")

        # 构建数据矩阵
        print("构建数据矩阵...")
        Z, Zp = data_matrices(X_td)

        # DMD-RRR
        print(f"执行 DMD-RRR (k_mode={args.k_mode})...")
        eigenvalues, modes, eigenvecs_right = dmd_rrr(Z, Zp, k_mode=args.k_mode)
        print(f"计算得到 {len(eigenvalues)} 个特征值")

        # 保存特征值
        base_filename = args.output_filename if args.output_filename else "spectrum"
        if base_filename.endswith('.png'):
            base_filename = base_filename[:-4]
        eigenvalues_path = os.path.join(output_dir, f"{base_filename}_eigenvalues.npy")
        np.save(eigenvalues_path, eigenvalues)
        print(f"特征值已保存至: {eigenvalues_path}")

        # 打印特征值统计信息
        print("\n特征值统计信息:")
        print(f"  数量: {len(eigenvalues)}")
        print(f"  模长范围: [{np.abs(eigenvalues).min():.4f}, {np.abs(eigenvalues).max():.4f}]")
        print(f"  实部范围: [{eigenvalues.real.min():.4f}, {eigenvalues.real.max():.4f}]")
        print(f"  虚部范围: [{eigenvalues.imag.min():.4f}, {eigenvalues.imag.max():.4f}]")
        print(f"  单位圆内特征值数量: {np.sum(np.abs(eigenvalues) < 1.0)}")
        print(f"  单位圆上特征值数量: {np.sum(np.abs(np.abs(eigenvalues) - 1.0) < 0.01)}")
        print(f"  单位圆外特征值数量: {np.sum(np.abs(eigenvalues) > 1.0)}")

        # 绘制谱图
        spectrum_path = os.path.join(output_dir, f"{base_filename}.png")
        scale_suffix = " (去尺度)" if args.normalize_scale else ""
        title = f"DMD-RRR 特征值谱 (n_delays={args.n_delays}, k_mode={args.k_mode}){scale_suffix}"
        plot_spectrum(eigenvalues, spectrum_path, title=title)

        print("\n分析完成！")


if __name__ == "__main__":
    main()
