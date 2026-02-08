#!/usr/bin/env python
"""
运行初始化微扰实验（正确答案 log-prob 版）

实验配置：
- 基础种子: 42（用于训练RNG和基础初始化）
- 微扰种子: 100-109（共10种微扰）
- weight_decay: 1, 2
- 对照组: 无微扰 (perturbation_seed=None)
- 共 2 (wd) x (1 + 10) (对照 + 微扰) = 22 个实验

微扰方式:
- 对每个参数 p，乘以 (1 + k * x)
- k = 0.001
- x ~ N(0, 1)
"""
import subprocess
import sys
import os
import shutil

# 实验配置
BASE_SEED = 42  # 基础种子（用于训练RNG和基础初始化）
PERTURBATION_SEEDS = list(range(100, 110))  # 100-109，共10种微扰
WD_VALUES = [1, 2]
PERTURBATION_SCALE = 0.001

# 基础参数（所有实验共用）
BASE_ARGS = [
    "--enable_fingerprint",
    "--fingerprint_source", "log_prob",
    "--probe_size", "100",
    "--n_quantiles", "19",
    "--record_every", "2",
    "--token_index", "-1",
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
OUTPUT_BASE = "logs/grokking_study/perturbation_correct_logprob"

def is_experiment_complete(logdir, min_snapshots=2500):
    """检查实验是否已完成"""
    fp_dir = os.path.join(logdir, "fingerprints")
    steps_file = os.path.join(fp_dir, "steps.npy")
    
    if not os.path.exists(steps_file):
        return False
    
    # 检查快照数量
    import glob
    snapshots = glob.glob(os.path.join(fp_dir, "snapshot_*.npy"))
    return len(snapshots) >= min_snapshots

def run_experiment(wd, perturbation_seed=None, force_overwrite=False, skip_completed=True):
    """
    运行单个实验
    
    :param wd: weight_decay 值
    :param perturbation_seed: 微扰种子，None 表示无微扰（对照组）
    :param force_overwrite: 是否覆盖旧目录
    :param skip_completed: 是否跳过已完成的实验
    """
    if perturbation_seed is None:
        exp_name = f"wd{wd}_baseline"
    else:
        exp_name = f"wd{wd}_perturb{perturbation_seed}"
    
    logdir = os.path.join(OUTPUT_BASE, exp_name)
    
    # 检查是否已完成
    if skip_completed and is_experiment_complete(logdir):
        print(f"[SKIP] {exp_name} 已完成，跳过")
        return True
    
    # 如果需要覆盖，删除旧目录
    if force_overwrite and os.path.exists(logdir):
        print(f"[OVERWRITE] 删除旧目录: {logdir}")
        try:
            shutil.rmtree(logdir)
        except (PermissionError, OSError) as e:
            print(f"  警告: 无法删除目录 {logdir}: {e}")
            print(f"  请手动删除或关闭可能正在使用该目录的程序")
            return False
    elif os.path.exists(logdir):
        # 删除不完整的实验目录
        print(f"[CLEANUP] 删除不完整的目录: {logdir}")
        try:
            shutil.rmtree(logdir)
        except (PermissionError, OSError) as e:
            print(f"  警告: 无法删除目录 {logdir}: {e}")
            # 继续运行，可能只是部分文件被锁定
    
    # 构建命令
    cmd = [
        sys.executable, "scripts/train.py",
        "--random_seed", str(BASE_SEED),
        "--logdir", logdir,
        "--weight_decay", str(wd),
    ] + BASE_ARGS
    
    # 添加微扰参数
    if perturbation_seed is not None:
        cmd.extend([
            "--perturbation_seed", str(perturbation_seed),
            "--perturbation_scale", str(PERTURBATION_SCALE),
        ])
    
    print(f"\n{'='*60}")
    print(f"运行实验: {exp_name}")
    print(f"  base_seed={BASE_SEED}, weight_decay={wd}")
    if perturbation_seed is not None:
        print(f"  perturbation_seed={perturbation_seed}, scale={PERTURBATION_SCALE}")
    else:
        print(f"  无微扰（对照组）")
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
    print("运行初始化微扰实验（正确答案 log-prob 版）")
    print("=" * 70)
    print(f"基础种子: {BASE_SEED}")
    print(f"微扰种子: {PERTURBATION_SEEDS}")
    print(f"weight_decay 值: {WD_VALUES}")
    print(f"微扰尺度: {PERTURBATION_SCALE}")
    print(f"总实验数: {len(WD_VALUES) * (1 + len(PERTURBATION_SEEDS))}")
    print(f"输出目录: {OUTPUT_BASE}")
    print("=" * 70)
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    results = []
    total = len(WD_VALUES) * (1 + len(PERTURBATION_SEEDS))
    current = 0
    
    # 由于中心化方法已修复，需要重做所有实验以确保使用正确的中心化
    # 设置 force_overwrite=True 来覆盖旧实验
    # 注意：如果遇到权限错误，请手动删除旧实验目录
    FORCE_REDO = False  # 设置为 True 以重做所有实验，False 则只运行未完成的实验
    
    for wd in WD_VALUES:
        # 先运行对照组（无微扰）
        current += 1
        print(f"\n[{current}/{total}] 运行对照组 wd={wd}")
        success = run_experiment(wd, perturbation_seed=None, force_overwrite=FORCE_REDO, skip_completed=not FORCE_REDO)
        results.append((wd, None, success))
        
        # 运行各微扰实验
        for perturb_seed in PERTURBATION_SEEDS:
            current += 1
            print(f"\n[{current}/{total}] 运行微扰实验 wd={wd}, perturb={perturb_seed}")
            success = run_experiment(wd, perturbation_seed=perturb_seed, force_overwrite=FORCE_REDO, skip_completed=not FORCE_REDO)
            results.append((wd, perturb_seed, success))
    
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
        for wd, perturb, success in results:
            if not success:
                print(f"  wd={wd}, perturbation_seed={perturb}")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
