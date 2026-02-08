#!/usr/bin/env python
"""
运行新的 grokking log-prob 实验（使用正确的中心化方式）

实验配置：
- 种子: 47-51（后5个种子）
- weight_decay: 1, 2
- 训练集: 40%
- 早停: 99% 延迟 1000 步
- 最大步数: 6000
- 中心化: 训练时不中心化，分析时使用跨时间步中心化
"""
import subprocess
import sys
import os
import shutil

# 实验配置
SEEDS = [47, 48, 49, 50, 51]  # 后5个种子
WD_VALUES = [1, 2]

# 基础参数（所有实验共用）
BASE_ARGS = [
    "--enable_fingerprint",
    "--fingerprint_source", "log_prob",
    "--probe_size", "100",
    "--n_quantiles", "19",
    "--record_every", "2",
    "--token_index", "-1",
    # 注意：不再使用 center_mode，训练时保存未中心化的原始快照
    "--fingerprint_seed", "42",
    "--d_model", "128",
    "--n_layers", "2",
    "--n_heads", "4",
    "--math_operator", "+",
    "--train_data_pct", "40",
    "--max_lr", "1e-3",
    "--gpu", "0",
    "--max_steps", "6000",
    "--early_stop_threshold", "99",
    "--early_stop_delay_steps", "1000",
]

# 输出目录
OUTPUT_BASE = "logs/grokking_study/grokking_logprob_new"

def is_experiment_complete(logdir, min_snapshots=2000):
    """检查实验是否已完成"""
    fp_dir = os.path.join(logdir, "fingerprints")
    steps_file = os.path.join(fp_dir, "steps.npy")
    
    if not os.path.exists(steps_file):
        return False
    
    # 检查快照数量
    import glob
    snapshots = glob.glob(os.path.join(fp_dir, "snapshot_*.npy"))
    return len(snapshots) >= min_snapshots

def run_experiment(seed, wd, force_overwrite=False, skip_completed=True):
    """
    运行单个实验
    
    :param seed: 随机种子
    :param wd: weight_decay 值
    :param force_overwrite: 是否覆盖旧目录
    :param skip_completed: 是否跳过已完成的实验
    """
    exp_name = f"seed{seed}_wd{wd}"
    logdir = os.path.join(OUTPUT_BASE, exp_name)
    
    # 检查是否已完成
    if skip_completed and is_experiment_complete(logdir):
        print(f"[SKIP] {exp_name} 已完成，跳过")
        return True
    
    # 如果需要覆盖，删除旧目录
    if force_overwrite and os.path.exists(logdir):
        print(f"[OVERWRITE] 删除旧目录: {logdir}")
        shutil.rmtree(logdir)
    elif os.path.exists(logdir):
        # 删除不完整的实验目录
        print(f"[CLEANUP] 删除不完整的目录: {logdir}")
        shutil.rmtree(logdir)
    
    # 构建命令
    cmd = [
        sys.executable, "scripts/train.py",
        "--random_seed", str(seed),
        "--logdir", logdir,
        "--weight_decay", str(wd),
    ] + BASE_ARGS
    
    print(f"\n{'='*60}")
    print(f"运行实验: {exp_name}")
    print(f"  seed={seed}, weight_decay={wd}")
    print(f"  logdir={logdir}")
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
    print("运行新的 grokking log-prob 实验（使用正确的中心化方式）")
    print("=" * 70)
    print(f"种子: {SEEDS} (后5个种子)")
    print(f"weight_decay 值: {WD_VALUES}")
    print(f"训练集: 40%")
    print(f"早停: 99% 延迟 1000 步")
    print(f"最大步数: 6000")
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
            print(f"\n[{current}/{total}] 运行实验 seed={seed}, wd={wd}")
            success = run_experiment(seed, wd, force_overwrite=False, skip_completed=True)
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
