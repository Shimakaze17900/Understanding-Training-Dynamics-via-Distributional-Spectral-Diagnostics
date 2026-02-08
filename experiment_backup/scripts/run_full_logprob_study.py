#!/usr/bin/env python
"""
运行完整的 log-prob fingerprint 研究实验
- 10 个种子 (42-51)
- 3 种 wd 设置 (0, 1, 2)
- 共 30 个实验
"""
import subprocess
import sys
import os

# 实验配置
SEEDS = list(range(42, 52))  # 42-51, 共10个种子
WD_VALUES = [0, 1, 2]

# 基础参数（所有实验共用）
BASE_ARGS = [
    "--enable_fingerprint",
    "--fingerprint_source", "log_prob",
    "--probe_size", "100",
    "--proj_dim", "100",
    "--n_quantiles", "19",
    "--record_every", "2",
    "--token_index", "-1",
    "--center_mode", "per_snapshot_mean",
    "--fingerprint_seed", "42",
    "--d_model", "128",
    "--n_layers", "2",
    "--n_heads", "4",
    "--math_operator", "+",
    "--train_data_pct", "40",
    "--max_lr", "1e-3",
    "--gpu", "0",
]

# 输出目录
OUTPUT_BASE = "logs/grokking_study/logprob_full_study"

def run_experiment(seed, wd):
    """运行单个实验"""
    if wd == 0:
        # 不带 wd 的实验：10000 步，无早停
        exp_name = f"seed{seed}_nowd"
        extra_args = [
            "--max_steps", "10000",
            "--weight_decay", "0",
        ]
    else:
        # 带 wd 的实验：6000 步，早停阈值 99%，延迟 1000 步
        exp_name = f"seed{seed}_wd{wd}"
        extra_args = [
            "--max_steps", "6000",
            "--weight_decay", str(wd),
            "--early_stop_threshold", "99",  # 99% 准确率触发早停
            "--early_stop_delay_steps", "1000",
        ]
    
    logdir = os.path.join(OUTPUT_BASE, exp_name)
    
    # 检查是否已完成
    fingerprint_dir = os.path.join(logdir, "fingerprints")
    if os.path.exists(fingerprint_dir):
        steps_file = os.path.join(fingerprint_dir, "steps.npy")
        if os.path.exists(steps_file):
            import numpy as np
            steps = np.load(steps_file)
            if len(steps) > 100:  # 至少有100个快照表示实验已完成
                print(f"[SKIP] {exp_name}: 已存在 {len(steps)} 个快照")
                return True
    
    # 构建命令
    cmd = [
        sys.executable, "scripts/train.py",
        "--random_seed", str(seed),
        "--logdir", logdir,
    ] + BASE_ARGS + extra_args
    
    print(f"\n{'='*60}")
    print(f"运行实验: {exp_name}")
    print(f"  seed={seed}, weight_decay={wd}")
    print(f"  logdir={logdir}")
    if wd > 0:
        print(f"  max_steps=6000, early_stop_threshold=99%, delay=1000")
    else:
        print(f"  max_steps=10000, no early stop")
    print(f"{'='*60}\n")
    
    # 运行
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode == 0:
        print(f"\n[SUCCESS] {exp_name} 完成")
        return True
    else:
        print(f"\n[FAILED] {exp_name} 失败，返回码: {result.returncode}")
        return False

def main():
    print("=" * 70)
    print("开始运行 log-prob fingerprint 完整研究")
    print(f"种子: {SEEDS}")
    print(f"weight_decay 值: {WD_VALUES}")
    print(f"总实验数: {len(SEEDS) * len(WD_VALUES)}")
    print(f"输出目录: {OUTPUT_BASE}")
    print("=" * 70)
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    results = []
    total = len(SEEDS) * len(WD_VALUES)
    current = 0
    
    for seed in SEEDS:
        for wd in WD_VALUES:
            current += 1
            print(f"\n[{current}/{total}] 开始实验 seed={seed}, wd={wd}")
            success = run_experiment(seed, wd)
            results.append((seed, wd, success))
    
    # 打印总结
    print(f"\n{'='*70}")
    print("实验总结:")
    print(f"{'='*70}")
    
    success_count = sum(1 for _, _, s in results if s)
    fail_count = len(results) - success_count
    
    print(f"成功: {success_count}/{len(results)}")
    print(f"失败: {fail_count}/{len(results)}")
    
    if fail_count > 0:
        print("\n失败的实验:")
        for seed, wd, success in results:
            if not success:
                print(f"  seed={seed}, wd={wd}")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
