# Multi-GPU/Multi-CPU Exact Gaussian Process Regression (GPR)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![GPyTorch](https://img.shields.io/badge/GPyTorch-1.11+-green.svg)](https://gpytorch.ai/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

High-performance Exact Gaussian Process Regression implementation with true parallel processing across multiple GPUs or CPU cores using PyTorch and GPyTorch. Designed for efficient compound coastal flooding prediction and other large-scale regression tasks.

## 🚀 Key Features

- **Flexible Hardware Acceleration**: Interactive selection of Multi-GPU, Single-GPU, Multi-CPU, or Single-CPU execution
- **True Parallel Processing**: Uses `ProcessPoolExecutor` for genuine parallelism across devices during both training and prediction
- **Adaptive Performance**: Automatic batch sizing and memory optimization based on dataset size
- **Comprehensive Grid Search**: Tests multiple kernels, optimizers, and hyperparameters with parallel cross-validation
- **Production Ready**: Robust error handling, model persistence, and detailed logging
- **HPC Compatible**: Platform-aware CSV reading and non-interactive plotting for batch job environments

## 📊 Performance Highlights

Tested on a real-world coastal flooding dataset (4.4M samples) with significant speedups compared to MATLAB CPU implementations:

- **Depending on Mode**: Up to 60-100x faster training and 10-60x faster prediction
- **Memory Efficient**: Handles datasets >1M samples with adaptive batching

## 🔧 Installation

### Prerequisites

```bash
# Core dependencies
pip install torch>=2.0.0 torchvision torchaudio
pip install gpytorch>=1.11
pip install scikit-learn pandas numpy matplotlib
pip install joblib tqdm
```

### Optional: CUDA Setup

For GPU acceleration, ensure CUDA-compatible PyTorch is installed:

```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## 📖 Usage

### Basic Usage

```bash
python ExactGPs_GPU.py
```

The script will interactively prompt you to select hardware configuration.

### Hardware Selection Options

#### GPU Mode (if GPUs detected):
```
1. Use ALL available GPUs (parallel training & prediction)
2. Use SINGLE GPU (sequential, no parallelism)
3. Select SPECIFIC GPUs (e.g., 1,2,5 to skip failing GPUs)
4. Use CPU instead (will prompt for core count)
```

#### CPU Mode (if no GPUs or option 4 selected):
```
Enter number of CPU cores (1-N):
  - 1 core: Sequential processing
  - N cores: Parallel processing (recommended)
```

### Environment Variable (HPC Batch Jobs)

For non-interactive execution:

```bash
# Select specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
python ExactGPs_GPU.py

# Use CPU only
export CUDA_VISIBLE_DEVICES=''
python ExactGPs_GPU.py
```

### Data Format

Prepare your data as CSV files:

- **Training inputs**: `Forcing_750sim_SF.csv` (N samples × M features)
- **Training targets**: `TWL_750sim_SF.csv` (N samples × 1)
- **Prediction inputs**: `Inputs_GPR_predict_500yr_sim_1.csv`

The script automatically handles:
- Degree-to-radian conversion for first 10 columns
- StandardScaler normalization
- Data validation (NaN/Inf checks)

## 🔬 Model Configuration

### Kernel Options

The script tests various kernel combinations:

```python
kernel_options = [
    # Simple kernels
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=n_features)),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_features)),
    
    # Composite kernels
    gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=n_features) + 
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_features)
    ),
]
```

### Optimizers

Supported optimizers with automatic hyperparameter tuning:
- Adam (default)
- SGD with momentum
- RMSprop
- AdamW
- Adagrad
- Adadelta
- LBFGS

### Training Parameters, e.g., 

```python
learning_rate_options = [0.1]
epoch_options = [200]
early_stopping_patience = 30  # Adjustable
early_stopping_delta = 1e-4
```

## 📁 Output Files

The script generates:

- `best_gpytorch_model_enhanced.pth` - Trained model with full configuration
- `X_scaler_enhanced.save` - Input feature scaler
- `y_scaler_enhanced.save` - Target variable scaler
- `GPR_Enhanced_RMSE_R2_by_SampleSize.csv` - Cross-validation results
- `GPR_Enhanced_Performance_Metrics.csv` - Detailed metrics for each configuration
- `GPR_Enhanced_Predictions_MultiGPU.csv` - Prediction results
- `Fig/GPR_Enhanced_RMSE_R2_by_SampleSize.png` - Performance visualization

## 🏗️ Architecture

### Multi-GPU Strategy

**NOT using DataParallel**: GPyTorch's lazy tensors are incompatible with DataParallel. Instead:

1. **Training**: Distribute 5-fold cross-validation folds across GPUs using `ProcessPoolExecutor`
   - Each fold runs in independent process with dedicated GPU
   - True parallel execution without device conflicts

2. **Prediction**: Split dataset into chunks, process on separate GPUs simultaneously
   - Each GPU handles ~1/N of total samples
   - Results concatenated after completion

### Process Architecture

```
Main Process
├── Fork Process 1 → GPU 0 (Fold 1)
├── Fork Process 2 → GPU 1 (Fold 2)
├── Fork Process 3 → GPU 2 (Fold 3)
├── Fork Process 4 → GPU 3 (Fold 4)
└── Fork Process 5 → GPU 0 (Fold 5)
```

### Key Technical Details

- **Multiprocessing start method**: `spawn` (required for CUDA compatibility)
- **Kernel serialization**: JSON-based for safe inter-process communication
- **Memory management**: Aggressive cleanup + adaptive batch sizing
- **Gradient clipping**: Max norm 1.0 for training stability
- **Learning rate scheduling**: ReduceLROnPlateau with patience=10

## 🎯 Example Workflow

```python
# 1. Load and preprocess data
X_ftf, y_ftf = load_and_validate_data()
X, y, X_scaler, y_scaler = preprocess_data(X_ftf, y_ftf)

# 2. Setup hardware (interactive or via environment variable)
# User selects: Multi-GPU, Single-GPU, or Multi-CPU

# 3. Grid search with parallel cross-validation
# Automatically distributes folds across selected hardware
for kernel, optimizer, lr, epochs in combinations:
    cv_results = parallel_cross_validation(...)

# 4. Train best model on full dataset
best_model = train_best_configuration(...)

# 5. Parallel prediction on new data
predictions = predict_with_enhanced_model_multi_gpu(...)
```

## 🔍 Troubleshooting

### Out of Memory Errors

**Solution**: The script automatically adjusts batch size based on dataset size:
- \>1M samples: batch_size = 100
- 100K-1M: batch_size = 500
- <100K: batch_size = 1000

For persistent issues:
```python
# Manually reduce batch size in predict function
predictions = predict_with_enhanced_model_multi_gpu(batch_size=50)
```

### GPU Device Conflicts

**Solution**: Use option 3 to exclude problematic GPUs:
```
Enter option: 3
GPU IDs: 0,1,3,4  # Skip GPU 2
```

### CUDA Out of Memory During Prediction

**Solution**: The script uses chunked processing. For very large datasets:
```python
# Reduce number of parallel workers
predictions = predict_with_enhanced_model_multi_gpu(num_workers=2)
```

### Windows OneDrive CSV Loading Issues

**Solution**: Script automatically uses Python engine on Windows:
```python
# Already handled internally
if platform.system() == 'Windows':
    pd.read_csv(..., engine='python')  # More robust
else:
    pd.read_csv(..., engine='c')  # Faster
```

### Lazy Tensor Thread Safety

**Solution**: Use `ProcessPoolExecutor` instead of `ThreadPoolExecutor`:
```python
# Correct (already implemented)
with ProcessPoolExecutor(max_workers=n_gpus) as executor:
    ...

# Incorrect (causes lazy tensor issues)
with ThreadPoolExecutor(max_workers=n_gpus) as executor:
    ...
```

## 📚 References

This implementation was developed for compound coastal flooding prediction:

1. Wang, Z., Leung, M., Mukhopadhyay, S., et al. (2024). "A hybrid statistical–dynamical framework for compound coastal flooding analysis." *Environmental Research Letters*, 20(1), 014005.

2. Wang, Z., Leung, M., Mukhopadhyay, S., et al. (2025). "Compound coastal flooding in San Francisco Bay under climate change." *npj Natural Hazards*, 2(1), 3.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Integration with other GP libraries

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

**Zhenqiang Wang**
- Email: zhenqiang.wang@oregonstate.edu
- Affiliation: Oregon State University

## 🙏 Acknowledgments

- GPyTorch team for excellent GPU-accelerated GP library
- PyTorch team for robust deep learning framework
- Coastal flooding research collaborators

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2024hybrid,
  title={A hybrid statistical--dynamical framework for compound coastal flooding analysis},
  author={Wang, Zhenqiang and Leung, Meredith and Mukhopadhyay, Sudarshana and Sunkara, Sai Veena and Steinschneider, Scott and Herman, Jonathan and Abellera, Marriah and Kucharski, John and Nederhoff, Kees and Ruggiero, Peter},
  journal={Environmental Research Letters},
  volume={20},
  number={1},
  pages={014005},
  year={2024},
  publisher={IOP Publishing}
}
@article{wang2025compound,
  title={Compound coastal flooding in San Francisco Bay under climate change},
  author={Wang, Zhenqiang and Leung, Meredith and Mukhopadhyay, Sudarshana and Sunkara, Sai Veena and Steinschneider, Scott and Herman, Jonathan and Abellera, Marriah and Kucharski, John and Ruggiero, Peter},
  journal={npj Natural Hazards},
  volume={2},
  number={1},
  pages={3},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
}
```

---

**Version**: 1.0.0  
**Last Updated**: September 2025  
**Python**: 3.8+  
**PyTorch**: 2.0+  
**GPyTorch**: 1.11+
