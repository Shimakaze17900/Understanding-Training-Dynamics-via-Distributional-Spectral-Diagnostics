# Grokking Experiments - Backup Repository

## Overview

This backup repository contains the essential experimental scripts and configuration files for studying the grokking phenomenon in Transformer models. This codebase is built upon the repository from the paper **"Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"** by Power et al. (2022) ([arXiv:2201.02177](https://arxiv.org/abs/2201.02177)).

## Experimental Setup

### Core Components

The experiments focus on analyzing the grokking phenomenon using:

1. **Transformer Architecture**: Small-scale Transformer models trained on modular arithmetic tasks
2. **Fingerprint Recording**: During training, snapshots of model representations are captured using quantile embeddings
3. **Spectral Analysis**: Dynamic Mode Decomposition with Reduced Rank Regression (DMD-RRR) is applied to analyze the training dynamics

### Key Experimental Parameters

- **Model Architecture**: 
  - `d_model`: 128
  - `n_layers`: 2
  - `n_heads`: 4
- **Training Configuration**:
  - Math operator: Addition (+)
  - Training data percentage: 40%
  - Maximum learning rate: 1e-3
  - Weight decay values: 0, 1, 2
  - Maximum training steps: 6000
  - Early stopping: 99% validation accuracy with 1000-step delay
- **Fingerprint Recording**:
  - Probe size: 100 samples
  - Number of quantiles: 19 (covering 0.05-0.95 range)
  - Recording frequency: Every 2 training steps
  - Token index: -1 (last token)
  - Fingerprint source: Log-probability distributions

## Experimental Objectives

1. **Understanding Grokking Dynamics**: Investigate how Transformer models transition from memorization to generalization during training
2. **Spectral Analysis**: Use DMD-RRR to extract dynamic modes and eigenvalues that characterize the training process
3. **Robustness Studies**: Examine the stability of training dynamics across different random seeds and weight decay values
4. **Perturbation Analysis**: Study how small initial perturbations affect the training trajectory and final generalization

## Repository Structure

```
experiment_backup/
├── grok/                          # Core library modules
│   ├── __init__.py
│   ├── data.py                    # Data loading and preprocessing
│   ├── training.py                # Training loop and model definition
│   ├── transformer.py             # Transformer architecture
│   ├── metrics.py                 # Evaluation metrics
│   ├── measure.py                 # Measurement utilities
│   ├── visualization.py           # Visualization tools
│   └── fingerprint/               # Fingerprint analysis modules
│       ├── __init__.py
│       ├── dmd_rrr.py             # DMD-RRR implementation
│       ├── quantile_embedding.py  # Quantile embedding computation
│       └── utils.py               # Utility functions
├── scripts/                        # Experimental scripts
│   ├── train.py                   # Main training script
│   ├── make_data.py               # Data generation script
│   ├── analyze_fingerprint.py     # Fingerprint analysis script
│   ├── run_grokking_logprob_new.py           # Run grokking experiments
│   ├── run_full_logprob_study.py             # Full experimental study
│   └── run_perturbation_experiments_correct_logprob.py  # Perturbation experiments
├── setup.py                        # Package installation configuration
└── README.md                       # This file
```

## Script Usage

### 1. Installation

First, install the package and its dependencies:

```bash
pip install -e .
```

### 2. Data Preparation

Generate the arithmetic dataset:

```bash
python scripts/make_data.py
```

### 3. Training with Fingerprint Recording

Train a model while recording fingerprint snapshots:

```bash
python scripts/train.py \
    --enable_fingerprint \
    --fingerprint_source log_prob \
    --probe_size 100 \
    --n_quantiles 19 \
    --record_every 2 \
    --token_index -1 \
    --fingerprint_seed 42 \
    --d_model 128 \
    --n_layers 2 \
    --n_heads 4 \
    --math_operator + \
    --train_data_pct 40 \
    --max_lr 1e-3 \
    --max_steps 6000 \
    --early_stop_threshold 99 \
    --early_stop_delay_steps 1000 \
    --logdir logs/my_experiment \
    --random_seed 42 \
    --gpu 0
```

**Key Parameters:**
- `--enable_fingerprint`: Enable fingerprint recording during training
- `--fingerprint_source`: Source of fingerprint data (`log_prob` for log-probability distributions)
- `--probe_size`: Number of probe samples to use
- `--n_quantiles`: Number of quantile points (default: 19)
- `--record_every`: Record snapshot every N training steps
- `--logdir`: Directory to save training logs and fingerprints

### 4. Running Batch Experiments

Run multiple experiments with different seeds and weight decay values:

```bash
python scripts/run_grokking_logprob_new.py
```

This script runs experiments for:
- Seeds: 47, 48, 49, 50, 51
- Weight decay: 1, 2

### 5. Running Perturbation Experiments

Study the effect of initial weight perturbations:

```bash
python scripts/run_perturbation_experiments_correct_logprob.py
```

### 6. Analyzing Fingerprints

After training, analyze the recorded fingerprints to extract DMD-RRR eigenvalues:

```bash
python scripts/analyze_fingerprint.py \
    --run_dir logs/my_experiment \
    --n_delays 32 \
    --k_mode 10 \
    --center_mode per_snapshot_mean
```

**Parameters:**
- `--run_dir`: Directory containing the `fingerprints/` subdirectory
- `--n_delays`: Time-delay embedding parameter (default: 32)
- `--k_mode`: DMD-RRR SVD truncation rank (default: 10)
- `--center_mode`: Centering mode (`per_snapshot_mean` or `relative_to_t0`)

The analysis generates:
- `eigenvalues.npy`: DMD eigenvalues (complex numbers)
- `spectrum.png` and `spectrum.pdf`: Eigenvalue spectrum plots

## Technical Details

### Quantile Embedding

The fingerprint system uses quantile embeddings to capture the distribution of model representations:
- For each training step, a fixed set of probe samples is selected
- Log-probability values for the correct answers are computed
- These values are converted to a quantile embedding vector using empirical quantiles
- The embedding dimension is determined by the number of quantiles (typically 19)

### DMD-RRR Analysis

Dynamic Mode Decomposition with Reduced Rank Regression:
1. **Time-delay embedding**: The snapshot sequence is converted to a Hankel matrix
2. **Data matrices**: Construct Z and Z' matrices for DMD analysis
3. **SVD truncation**: Apply reduced-rank regression with rank k_mode
4. **Eigenvalue extraction**: Compute DMD eigenvalues that characterize the dynamic modes

The eigenvalues are plotted in the complex plane, with the unit circle indicating stability boundaries.

## Output Structure

Training outputs are saved in the specified `logdir`:

```
logdir/
├── fingerprints/
│   ├── config.json              # Fingerprint configuration
│   ├── probe_indices.npy        # Indices of probe samples
│   ├── snapshot_tXXXXXX.npy     # Snapshot at step XXXXXX
│   ├── steps.npy                # Training step numbers
│   ├── eigenvalues.npy          # DMD eigenvalues (after analysis)
│   └── spectrum.png/pdf         # Spectrum plots (after analysis)
└── lightning_logs/              # PyTorch Lightning logs
```

## Dependencies

Key dependencies (see `setup.py` for full list):
- PyTorch
- PyTorch Lightning
- NumPy
- SciPy
- Matplotlib

## Citation

If you use this codebase, please cite the original paper:

```bibtex
@article{power2022grokking,
  title={Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets},
  author={Power, Alethea and Burda, Yuri and Edwards, Harri and Babuschkin, Igor and Misra, Vedant},
  journal={arXiv preprint arXiv:2201.02177},
  year={2022}
}
```

## Notes

- This backup repository contains only essential scripts and configuration files
- Temporary analysis scripts and monitoring tools are excluded
- All experimental results and logs are stored separately in the `logs/` directory
- The codebase is designed for research purposes and may require modifications for different experimental setups
