# Distribution Snapshot FCN Experiment - Setup and Usage Guide

This folder contains backups of all critical scripts for the **Distribution Snapshot Fully Connected Neural Network (FCN) experiment**. This experiment uses a novel distributional snapshot method to study training dynamics of fully connected neural networks trained on MNIST, enabling comparison across different network widths using Koopman operator theory.

## Table of Contents

1. [Experiment Overview](#experiment-overview)
2. [Key Methodology: Distribution Snapshot Method](#key-methodology-distribution-snapshot-method)
3. [Script Organization](#script-organization)
4. [Experimental Setup](#experimental-setup)
5. [Usage Instructions](#usage-instructions)
6. [Output Structure](#output-structure)
7. [Dependencies and Requirements](#dependencies-and-requirements)
8. [Technical Details](#technical-details)

---

## Experiment Overview

### Research Objective

This experiment studies how network width affects training dynamics in fully connected neural networks using a **distributional snapshot approach**. Unlike traditional methods that directly analyze high-dimensional weight vectors, this approach:

1. **Compresses high-dimensional weights** into low-dimensional distributional representations (19 quantiles)
2. **Enables cross-width comparison** by using the same representation dimension regardless of network width
3. **Extracts Koopman eigenvalues** using Dynamic Mode Decomposition (DMD) to characterize training dynamics
4. **Identifies equivalent training dynamics** across different network widths

### Key Innovation: Distribution Snapshot Method

The distribution snapshot method addresses a fundamental challenge: **how to compare training dynamics of networks with different widths** (e.g., h=5 vs h=1024), where the weight dimensions differ dramatically.

**Solution**: Instead of analyzing raw weights, we:
- Extract **19 quantiles** (0.05 to 0.95) from the weight distribution at each time step
- Create a **distributional snapshot** of shape (T, 19) regardless of network width
- Apply **cross-step centering** to remove mean shifts
- Perform **DMD analysis** on these low-dimensional snapshots

This approach provides:
- **Dimensionality invariance**: Same representation dimension (19) for all network widths
- **Permutation invariance**: Focuses on distribution rather than individual weight values
- **Computational efficiency**: 19-dimensional data vs. 10×h-dimensional raw weights

---

## Script Organization

The `Distribution_Snapshot_FCN/` folder contains the following core scripts:

### Core Training and Analysis Scripts

1. **`train_distribution_snapshot.py`** - Main training script
   - Trains FCNs on MNIST with varying widths (h=5, 40, 256, 1024)
   - Records output layer weights at each training step
   - Saves weight trajectories for subsequent analysis

2. **`combined_dmd_analysis.py`** - Cross-seed stacked DMD analysis
   - Loads weight trajectories from all seeds
   - Constructs distribution snapshots (19 quantiles)
   - Stacks snapshots across seeds for robust DMD analysis
   - Computes Koopman eigenvalues and reconstruction errors
   - Generates spectral plots

3. **`analyze_distribution_snapshot.py`** - Single-seed analysis
   - Performs DMD analysis on individual seed trajectories
   - Useful for seed-specific analysis and debugging

### Utility Modules

4. **`dynamic_mode_decomposition.py`** - DMD algorithm implementation
   - `time_delay()`: Time delay embedding
   - `data_matrices()`: Construct data matrices for DMD
   - `dmd()`: Perform DMD decomposition
   - `dim_reduction()`: Dimensionality reduction

5. **`wasserstein_distance.py`** - Wasserstein distance computation
   - `wass_dist()`: Compute Wasserstein distance between distributions
   - Used for comparing Koopman eigenvalue spectra

---

## Experimental Setup

### Network Architecture

```
Input Layer:  28×28 = 784 dimensions (flattened MNIST images)
    ↓
Hidden Layer: Linear(784, h) + ReLU activation
    ↓
Output Layer: Linear(h, 10) + LogSoftmax
```

**Key Details:**
- No bias terms in any layer
- Weight initialization: PyTorch default × 1.0
- Activation: ReLU (can be changed to GELU)

### Training Configuration

**Network Widths (h):**
- h = 5, 40, 256, 1024

**Random Seeds:**
- 25 seeds: [2, 3, 6, 7, 11, 16, 17, 22, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
- 1 initialization per seed

**Training Parameters:**
- **Optimizer**: SGD
- **Learning Rate**: 0.1
- **Batch Size**: 60
- **Epochs**: 1
- **Training Steps**: ~1000 (one per minibatch)
- **Dataset**: MNIST (60,000 training samples, 10,000 test samples)

**Data Preprocessing:**
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### Distribution Snapshot Parameters

- **Number of Quantiles**: 19
- **Quantile Range**: 0.05 to 0.95 (uniformly spaced)
- **Quantile Values**: `np.linspace(0.05, 0.95, 19)`
- **Snapshot Dimension**: (T, 19) where T = 1000 (training steps)

### DMD Parameters

- **Time Delays (n_delays)**: 32
- **DMD Modes (k_mode)**: 10
- **Effective Rank Thresholds**:
  - Fixed threshold: 1e-6
  - Energy threshold: 99%

---

## Usage Instructions

### Step 1: Training Networks

Run the training script to generate weight trajectories:

```bash
cd Distribution_Snapshot_FCN
python train_distribution_snapshot.py
```

**What it does:**
1. Trains FCNs with widths h=5, 40, 256, 1024
2. For each width, trains 25 networks (one per seed)
3. Records output layer weights at each training step
4. Saves weight trajectories to `Results/` folder

**Output Files:**
- `SGD_MNIST_h={h}_seed{seed}_initialization0_weights.npy` - Weight trajectories (shape: T × 10×h)
- `SGD_MNIST_h={h}_seed{seed}_initialization0_Train_Loss.npy` - Training loss
- `SGD_MNIST_h={h}_seed{seed}_initialization0_Loss.npy` - Test loss
- `SGD_MNIST_h={h}_seed{seed}_initialization0_Test_Accuracy.npy` - Test accuracy

**Note:** Update file paths in the script to match your local directory structure.

### Step 2: Cross-Seed Stacked DMD Analysis

After training, run the combined DMD analysis:

```bash
python combined_dmd_analysis.py
```

**What it does:**
1. Loads weight trajectories from all seeds
2. Constructs distribution snapshots (19 quantiles) for each seed
3. Applies cross-step centering
4. Stacks snapshots from all seeds (shape: 25×T × 19)
5. Performs DMD analysis on stacked data
6. Computes reconstruction errors and effective ranks at intervals (every 250 steps)
7. Generates Koopman eigenvalue spectra plots
8. Saves results to `combined_analysis/` folder

**Key Processing Steps:**
```python
# 1. Load weights (T × 10×h)
W = np.load('weights.npy')

# 2. Construct distribution snapshots (T × 19)
snapshots = construct_distribution_snapshots(W, quantiles)

# 3. Cross-step centering
snapshots_centered = cross_step_centering(snapshots)

# 4. Stack across seeds (25×T × 19)
combined_snapshots = np.vstack([snapshots_seed1, ..., snapshots_seed25])

# 5. Time delay embedding
X_delayed = time_delay(combined_snapshots, n_delays=32)

# 6. DMD analysis
eigenvalues, modes = dmd(Z.T, Z_prime.T, k=10)
```

**Output Files:**
- `Koopman_eigenvalues_combined_h={h}.npy` - Koopman eigenvalues
- `DMD_reconstruction_errors_combined_h={h}.npy` - Reconstruction errors
- `DMD_effective_ranks_fixed_combined_h={h}.npy` - Effective ranks (fixed threshold)
- `DMD_effective_ranks_energy_combined_h={h}.npy` - Effective ranks (energy threshold)
- `DMD_S_max_combined_h={h}.npy` - Maximum singular values
- `DMD_S_min_combined_h={h}.npy` - Minimum singular values
- `Koopman_eigenvalues_combined.png` - Spectral plot
- `DMD_combined_analysis.md` - Analysis report

### Step 3: Single-Seed Analysis (Optional)

For seed-specific analysis:

```bash
python analyze_distribution_snapshot.py
```

**What it does:**
- Performs DMD analysis on individual seed trajectories
- Useful for understanding seed-to-seed variability
- Generates seed-specific visualizations

---

## Output Structure

### Directory Structure

```
Results/
├── SGD_MNIST_h=5_seed{seed}_initialization0_weights.npy
├── SGD_MNIST_h=5_seed{seed}_initialization0_Train_Loss.npy
├── SGD_MNIST_h=5_seed{seed}_initialization0_Loss.npy
├── SGD_MNIST_h=5_seed{seed}_initialization0_Test_Accuracy.npy
├── SGD_MNIST_h=40_seed{seed}_initialization0_weights.npy
├── ...
└── SGD_MNIST_h=1024_seed{seed}_initialization0_weights.npy

combined_analysis/
├── Koopman_eigenvalues_combined_h=5.npy
├── Koopman_eigenvalues_combined_h=40.npy
├── Koopman_eigenvalues_combined_h=256.npy
├── Koopman_eigenvalues_combined_h=1024.npy
├── DMD_reconstruction_errors_combined_h={h}.npy
├── DMD_effective_ranks_fixed_combined_h={h}.npy
├── DMD_effective_ranks_energy_combined_h={h}.npy
├── DMD_S_max_combined_h={h}.npy
├── DMD_S_min_combined_h={h}.npy
├── Koopman_eigenvalues_combined.png
└── DMD_combined_analysis.md

Figures/
└── (generated visualization plots)
```

### Key Output Files

1. **Weight Trajectories** (`*_weights.npy`)
   - Shape: (T, 10×h) where T=1000, h=network width
   - Raw output layer weights at each training step

2. **Koopman Eigenvalues** (`Koopman_eigenvalues_combined_h={h}.npy`)
   - Complex-valued array of eigenvalues
   - Characterizes linearized training dynamics
   - Real part: growth/decay; Imaginary part: oscillations

3. **Reconstruction Errors** (`DMD_reconstruction_errors_combined_h={h}.npy`)
   - Normalized Frobenius norm of reconstruction error
   - Computed at intervals (every 250 steps)
   - Measures DMD model quality

4. **Effective Ranks** (`DMD_effective_ranks_*_combined_h={h}.npy`)
   - Fixed threshold: number of singular values > 1e-6
   - Energy threshold: number of modes capturing 99% energy
   - Measures intrinsic dimensionality

5. **Spectral Plot** (`Koopman_eigenvalues_combined.png`)
   - Visualization of eigenvalues in complex plane
   - Shows unit circle reference (|λ|=1)
   - Color-coded by network width

---

## Dependencies and Requirements

### Required Python Packages

```python
# Core scientific computing
numpy >= 1.19.0
scipy >= 1.5.0

# Deep learning
torch >= 1.8.0
torchvision >= 0.9.0

# Visualization
matplotlib >= 3.3.0

# Utilities
tqdm  # Progress bars
```

### Installation

```bash
pip install numpy scipy torch torchvision matplotlib tqdm
```

### System Requirements

- **Python**: 3.7 or higher
- **Memory**: 
  - Training: ~4-8 GB RAM (depending on network width)
  - Analysis: ~8-16 GB RAM (for stacked data: 25×1000×19)
- **Storage**: 
  - Weight trajectories: ~100-500 MB per network width
  - Total: ~1-2 GB for all experiments
- **GPU**: Optional but recommended for training (CUDA-compatible)

---

## Technical Details

### Distribution Snapshot Construction

**Algorithm:**
```python
def construct_distribution_snapshots(weights, quantiles):
    """
    Convert high-dimensional weights to distribution snapshots.
    
    Input: weights (T, D) where D = 10×h
    Output: snapshots (T, 19) where 19 = number of quantiles
    """
    T, D = weights.shape
    n_quantiles = len(quantiles)
    snapshots = np.zeros((T, n_quantiles))
    
    for t in range(T):
        # Extract quantiles from weight distribution at time t
        snapshots[t, :] = np.quantile(weights[t, :], quantiles)
    
    return snapshots
```

**Cross-Step Centering:**
```python
def cross_step_centering(snapshots):
    """
    Remove mean shift from each quantile across time steps.
    
    This ensures that DMD focuses on dynamics rather than
    absolute distribution values.
    """
    means = np.mean(snapshots, axis=0, keepdims=True)
    centered = snapshots - means
    return centered
```

### DMD Analysis Workflow

1. **Time Delay Embedding**
   - Input: (T, 19) snapshots
   - Output: (T-n_delays, (n_delays+1)×19) delayed data
   - Purpose: Capture temporal correlations

2. **Data Matrix Construction**
   - Z: First T-1 time steps
   - Z_prime: Last T-1 time steps (shifted by 1)
   - Purpose: Set up linear dynamics: Z_prime ≈ A @ Z

3. **DMD Decomposition**
   - SVD: Z = U @ S @ Vh
   - Truncate to k modes
   - Build Koopman operator: A = U.T @ Z_prime @ V @ inv(S)
   - Eigenvalue decomposition: A @ v = λ @ v
   - DMD modes: modes = U @ v

4. **Reconstruction Error**
   - Predict Z_prime using DMD modes and eigenvalues
   - Compute normalized Frobenius norm error
   - Measures how well DMD captures dynamics

### Cross-Seed Stacking Strategy

**Purpose:**
- Increase data volume for robust DMD analysis
- Capture common dynamics across different random initializations
- Reduce sensitivity to individual seed variations

**Implementation:**
```python
# Stack snapshots from all seeds
combined_snapshots = np.vstack([
    snapshots_seed1,   # (1000, 19)
    snapshots_seed2,   # (1000, 19)
    ...
    snapshots_seed25   # (1000, 19)
])
# Result: (25000, 19) = (25×1000, 19)
```

**Benefits:**
- More stable DMD eigenvalues
- Better statistical power
- Identifies universal training dynamics

### Interpreting Koopman Eigenvalues

**Complex Plane Visualization:**
- **Unit Circle (|λ|=1)**: Reference for stability
- **Real Part > 1**: Unstable/growing modes
- **Real Part < 1**: Stable/decaying modes
- **Real Part ≈ 1**: Dominant persistent modes
- **Imaginary Part ≠ 0**: Oscillatory modes

**Training Dynamics Insights:**
- Eigenvalues near unit circle: Long-term training behavior
- Eigenvalues inside unit circle: Transient dynamics
- Eigenvalues outside unit circle: Unstable dynamics (rare)

### Comparison Across Network Widths

The distribution snapshot method enables direct comparison:

1. **Same Representation Dimension**: All widths use 19 quantiles
2. **Wasserstein Distance**: Compare eigenvalue spectra using optimal transport
3. **Spectral Similarity**: Networks with similar eigenvalues have equivalent dynamics

**Key Finding**: Networks of different widths can exhibit equivalent training dynamics (similar Koopman eigenvalue spectra) when analyzed through distribution snapshots.

---

## Best Practices and Tips

1. **Path Configuration**: Update file paths in scripts to match your local directory structure.

2. **Memory Management**: 
   - For large networks (h=1024), consider processing in batches
   - Stacked data (25×1000×19) requires ~4 MB, manageable for most systems

3. **GPU Usage**: 
   - Training benefits significantly from GPU acceleration
   - DMD analysis is CPU-based (NumPy/SciPy)

4. **Parameter Tuning**:
   - `n_delays=32`: Balance between temporal correlation and computational cost
   - `k_mode=10`: Retain dominant modes, avoid noise
   - `19 quantiles`: Balance between information retention and dimensionality

5. **Reproducibility**:
   - Use fixed random seeds (provided in scripts)
   - Set PyTorch random seed: `torch.manual_seed(seed)`
   - Set NumPy random seed: `np.random.seed(seed)`

6. **Debugging**:
   - Start with small networks (h=5) for faster iteration
   - Use `analyze_distribution_snapshot.py` for single-seed analysis
   - Check weight trajectory shapes and ranges

7. **Visualization**:
   - Spectral plots show eigenvalues in complex plane
   - Compare across widths to identify equivalent dynamics
   - Check reconstruction errors to validate DMD quality

---

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce number of seeds or network width
   - Process data in batches
   - Use CPU instead of GPU for analysis

2. **File Not Found Errors**
   - Check file paths in scripts
   - Ensure training completed successfully
   - Verify Results/ folder structure

3. **DMD Reconstruction Errors Too High**
   - Increase `n_delays` for better temporal modeling
   - Increase `k_mode` to retain more modes
   - Check data quality (no NaN or Inf values)

4. **Inconsistent Results Across Seeds**
   - This is expected due to random initialization
   - Use cross-seed stacking for robust analysis
   - Check if specific seeds are outliers

---

## Citation

If you use this code, please cite:

```
@article{redman2024identifying,
  title={Identifying Equivalent Training Dynamics},
  author={Redman, William and others},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

---

## Contact

For questions about the distribution snapshot method or Koopman operator analysis, contact: wredman4@gmail.com

---

**Last Updated**: 2024  
**Version**: 2.0 (Distribution Snapshot Method)
