# -*- coding: utf-8 -*-
"""
Created on 9/16, 2025

@author: Zhenqiang Wang; Email: zhenqiang.wang@oregonstate.edu

Multi-GPU/Multi-CPU Exact Gaussian Process Regression (GPR) with Parallel Cross-Validation
===========================================================================================

This script implements GPU/CPU-accelerated Exact GPR using GPyTorch and PyTorch with 
true parallel processing across multiple GPUs or CPU cores for both training and prediction.
Supports flexible hardware selection: Multi-GPU, Single-GPU, Multi-CPU, or Single-CPU.

It is developed as an alternative to the MATLAB version running on CPU cores [1,2] to speed up the 
GPR trainings and predictions on large datasets for efficient compound coastal flooding prediction.
The script is tested on a real-world dataset (i.e., 500-year hourly flood drivers and associated total 
water level time series with 4382928 samples [1,2]) with flexible hardware selection options including
Multi-GPU, Single-GPU, Multi-CPU, or Single-CPU. It achieves significant speedups compared to the 
MATLAB CPU version while maintaining comparable accuracy, especially on GPU-accelerated hardware. 
It may be used for other regression tasks beyond coastal flooding prediction.

References:
[1] Wang, Z., Leung, M., Mukhopadhyay, S., Sunkara, S.V., Steinschneider, S., Herman, J., Abellera, M., 
    Kucharski, J., Nederhoff, K. and Ruggiero, P., 2024. A hybrid statistical–dynamical framework for 
    compound coastal flooding analysis. Environmental Research Letters, 20(1), p.014005.
[2] Wang, Z., Leung, M., Mukhopadhyay, S., Sunkara, S.V., Steinschneider, S., Herman, J., Abellera, M., 
    Kucharski, J. and Ruggiero, P., 2025. Compound coastal flooding in San Francisco Bay under climate change. 
    npj Natural Hazards, 2(1), p.3.

See full documentation below for detailed usage, troubleshooting, and examples.

Key Architecture:
-----------------
- **Unified Code Base**: Single implementation for both GPU and CPU modes - automatically 
  adapts based on available hardware
- **ProcessPoolExecutor**: Creates separate Python processes (not threads) for each
  CV fold and prediction chunk, enabling true parallel processing across multiple devices
- **Process-per-Device**: Each fold/chunk runs in its own process on a dedicated GPU/CPU with
  independent memory space for true parallelism without device conflicts
- **Kernel Serialization**: Kernels are serialized/deserialized for safe inter-process
  communication without device conflicts
- **Adaptive Batch Sizing**: Automatically adjusts batch size based on dataset size
  (100 for >1M samples, 500 for 100K-1M, 1000 for <100K)
- **Platform-Aware CSV Reading**: Fast C engine on Linux/GPU, robust Python engine on Windows/CPU
- **Single User Prompt**: Hardware selection asked once, used for both training and prediction

Main Features:
--------------
1. **Parallel Training**: 5-fold CV runs simultaneously on multiple GPUs/CPUs
2. **Parallel Prediction**: Distributed batch prediction across all GPUs/CPUs
3. **Flexible Hardware Selection**: Interactive GPU/CPU selection at runtime
4. **Grid Search**: Tests multiple kernels, optimizers, and hyperparameters
5. **Memory Management**: Automatic GPU memory optimization and cleanup
6. **Error Handling**: Robust error catching with fallback to sequential execution
7. **Model Persistence**: Save/load trained models with full configuration
8. **Performance Monitoring**: Comprehensive logging and timing statistics
9. **Training Stability**: Gradient clipping, learning rate scheduling, early stopping

Hardware Selection (Interactive):
---------------------------------
**GPU Mode (Options 1-4):**
  1. Use ALL GPUs (parallel training & prediction)
  2. Use SINGLE GPU (sequential, no parallelism)
  3. Select SPECIFIC GPUs (e.g., 1,2,5 to skip failing GPUs)
  4. Use CPU instead (will prompt for number of cores)

**CPU Mode (Direct Input):**
  Enter number of CPU cores (1-N):
    - 1 core: Sequential processing (slowest)
    - N cores: Parallel processing (recommended)
    - Same setting used for both training and prediction

Technical Details:
------------------
- Multiprocessing start method: 'spawn' (required for CUDA/CPU compatibility)
- Data passing: NumPy arrays + state_dicts (picklable objects only)
- Kernel handling: JSON-serializable kernel specifications reconstructed per device
- No DataParallel: GPyTorch's lazy tensors incompatible with DataParallel
- Memory optimization: Aggressive cleanup + adaptive batch sizing
- CSV Reading: C engine (Linux/GPU, fast) vs Python engine (Windows/CPU, robust)

Usage Examples:
---------------
# Interactive - GPU Mode (if GPUs available)
python ExactGPs_MGPU_SF.py
# Prompts: "Enter option (1/2/3/4)"
# Choose: 1 (all GPUs) / 2 (single GPU) / 3 (specific GPUs) / 4 (use CPU)

# Interactive - CPU Mode (if NO GPUs or option 4 selected)
python ExactGPs_MGPU_SF.py
# Prompts: "Number of CPU cores [default: N]"
# Choose: 1 (sequential) / N (parallel on N cores)

# Non-interactive - Environment variable (HPC batch jobs)
export CUDA_VISIBLE_DEVICES=1,2,5,6,7
python ExactGPs_MGPU_SF.py
# Script detects GPUs and uses them automatically

"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import gpytorch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import os
import matplotlib
# Set non-interactive backend for HPC environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
import joblib
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
from tqdm import tqdm
import gc

# CRITICAL: Set multiprocessing start method BEFORE any CUDA operations
# 'spawn' is required for CUDA to work properly with ProcessPoolExecutor
# This MUST be at the very top, before any other torch/cuda operations
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
# Setup enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# give a fixed seed for full reproducibility
seed_id = 42
torch.manual_seed(seed_id)
np.random.seed(seed_id)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

################################################################################################################
# Multi-GPU Manager and Performance Optimization
################################################################################################################

class MultiGPUManager:
    """Enhanced manager for multi-GPU operations and optimization"""
    
    def __init__(self):
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.is_distributed = False
        self.local_rank = 0
        self.world_size = 1
        
        logger.info(f"MultiGPU Manager initialized: {self.device_count} GPUs available")
        
    def setup_multi_gpu(self, use_distributed=False):
        """Setup multi-GPU environment with enhanced error handling"""
        try:
            if self.device_count > 1:
                if use_distributed:
                    # Initialize distributed training
                    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                        self.local_rank = int(os.environ['LOCAL_RANK'])
                        self.world_size = int(os.environ['WORLD_SIZE'])
                        torch.cuda.set_device(self.local_rank)
                        dist.init_process_group(backend='nccl')
                        self.is_distributed = True
                        logger.info(f"Initialized distributed training on {self.device_count} GPUs")
                    else:
                        logger.warning("Distributed training requested but environment not set up")
                else:
                    logger.info(f"Using DataParallel with {self.device_count} GPUs")
            elif self.device_count == 1:
                logger.info("Single GPU available")
            else:
                logger.warning("No CUDA GPUs available, using CPU")
            
            return self.device_count > 0
        except Exception as e:
            logger.error(f"Error setting up multi-GPU: {e}")
            return False
    
    def wrap_model_for_multi_gpu(self, model, use_distributed=False):
        """Wrap model for multi-GPU training with enhanced error handling"""
        try:
            if self.device_count > 1:
                if use_distributed and self.is_distributed:
                    model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[self.local_rank], output_device=self.local_rank,
                        find_unused_parameters=True
                    )
                    logger.info("Model wrapped with DistributedDataParallel")
                else:
                    model = torch.nn.DataParallel(model)
                    logger.info("Model wrapped with DataParallel")
            return model
        except Exception as e:
            logger.error(f"Error wrapping model for multi-GPU: {e}")
            return model
    
    def optimize_memory(self):
        """Enhanced GPU memory optimization"""
        if torch.cuda.is_available():
            try:
                for i in range(self.device_count):
                    try:
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    except RuntimeError as e:
                        # Skip if device is busy
                        logger.debug(f"Could not optimize memory on GPU {i}: {e}")
                        continue
                # Enable memory efficient attention if available
                if hasattr(torch.backends.cuda, 'enable_memory_efficient_sdp'):
                    torch.backends.cuda.enable_memory_efficient_sdp(True)
            except Exception as e:
                logger.debug(f"Error during memory optimization: {e}")
    
    def get_memory_info(self):
        """Get GPU memory information"""
        if torch.cuda.is_available():
            memory_info = []
            for i in range(self.device_count):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                cached = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                memory_info.append({
                    'gpu_id': i,
                    'allocated_gb': allocated,
                    'cached_gb': cached,
                    'total_gb': total,
                    'free_gb': total - allocated
                })
            return memory_info
        return []
    
    def distribute_batch_across_gpus(self, data, batch_size=None):
        """Distribute batch across multiple GPUs for prediction"""
        if not torch.cuda.is_available() or self.device_count <= 1:
            return [data]
        
        if batch_size is None:
            batch_size = len(data) // self.device_count
            
        chunks = []
        for i in range(self.device_count):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data))
            if start_idx < len(data):
                chunk = data[start_idx:end_idx].to(f'cuda:{i}')
                chunks.append((chunk, i))
        
        return chunks

gpu_manager = MultiGPUManager()

################################################################################################################
# ExactGPModel class with enhanced features
################################################################################################################

def _set_constraints_and_initial_values(kernel):
    """Enhanced constraint setting with better numerical stability"""
    if hasattr(kernel, "base_kernel"):
        _set_constraints_and_initial_values(kernel.base_kernel)
    if isinstance(kernel, (gpytorch.kernels.AdditiveKernel, gpytorch.kernels.ProductKernel)):
        for k in kernel.kernels:
            _set_constraints_and_initial_values(k)
    if hasattr(kernel, "raw_lengthscale"):
        kernel.lengthscale = 1.0
        kernel.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(0.01, 50.0))
    if hasattr(kernel, "raw_outputscale"):
        kernel.outputscale = 1.0
        kernel.register_constraint("raw_outputscale", gpytorch.constraints.Interval(0.001, 100.0))
    if hasattr(kernel, "raw_noise"):
        kernel.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-6, 1e-2))
        
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, set_init=True):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        if set_init:
            _set_constraints_and_initial_values(self.covar_module)
        
        # Enable gradient checkpointing for large datasets
        self._use_checkpointing = train_x.size(0) > 1000

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

################################################################################################################
# Enhanced data loading with validation
################################################################################################################
def load_and_validate_data():
    """Enhanced data loading with comprehensive validation"""
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        logger.info("Loading training data...")
        
        try:
            train_x_df = pd.read_csv('Forcing_750sim_SF.csv', header=None)
            train_y_df = pd.read_csv('TWL_750sim_SF.csv', header=None)
        except FileNotFoundError as e:
            logger.error(f"Data files not found: {e}")
            logger.warning("Creating synthetic data for testing...")
            n_samples, n_features = 750, 15
            train_x_df = pd.DataFrame(np.random.randn(n_samples, n_features))
            train_y_df = pd.DataFrame(np.random.randn(n_samples, 1))
        
        numpy_array_x = train_x_df.values
        X_ft = torch.from_numpy(numpy_array_x)
        X_ftf = X_ft.to(torch.float32)

        numpy_array_y = train_y_df.values
        y_ft = torch.from_numpy(numpy_array_y)
        y_ftf = y_ft.reshape(-1,1).to(torch.float32)
        
        # Enhanced data validation
        assert not torch.isnan(X_ftf).any(), "NaNs found in input data"
        assert not torch.isnan(y_ftf).any(), "NaNs found in target data"
        assert not torch.isinf(X_ftf).any(), "Infs found in input data"
        assert not torch.isinf(y_ftf).any(), "Infs found in target data"
        
        logger.info(f"Data loaded successfully: X shape {X_ftf.shape}, y shape {y_ftf.shape}")
        return X_ftf, y_ftf
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(X_ftf, y_ftf):
    """Enhanced data preprocessing with validation"""
    logger.info("Preprocessing data...")
    
    degree_columns = [i for i in range(min(10, X_ftf.shape[1]))]
    X_ftf_np = X_ftf.numpy() if isinstance(X_ftf, torch.Tensor) else X_ftf
    X_ftf_np[:, degree_columns] = np.deg2rad(X_ftf_np[:, degree_columns])
    X_ftf = torch.from_numpy(X_ftf_np)

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X = X_scaler.fit_transform(X_ftf)
    y = y_scaler.fit_transform(y_ftf.reshape(-1,1)).ravel()

    assert not np.isnan(X).any() and not np.isnan(y).any(), "NaNs in normalized data!"
    assert not np.isinf(X).any() and not np.isinf(y).any(), "Infs in normalized data!"
    
    unique_rows = np.unique(X, axis=0).shape[0]
    if unique_rows != X.shape[0]:
        logger.warning(f"Found {X.shape[0] - unique_rows} duplicate rows in X!")
    
    logger.info("Data preprocessing completed successfully")
    return X, y, X_scaler, y_scaler

################################################################################################################
# Enhanced training with multi-GPU support
################################################################################################################
# Define training parameters such as learning rates and epoch counts to try
# LBFGS with a high learning rate can cause instability and Reduce Learning Rate may be needed (e.g., 0.01).
# learning_rate_options = [0.01, 0.05, 0.1]
learning_rate_options = [0.1]
# epoch_options = [50, 100, 200]
epoch_options = [200]
# early_stopping_patience = 30  # Stop if loss doesn't improve for this many epochs, 10-30 even larger
# effectively disable early stopping for grid search
max_epochs = max(epoch_options)
early_stopping_patience = max_epochs + 1 
early_stopping_delta = 1e-4   # Minimum change to count as improvement

def train_and_eval_enhanced(train_x, train_y, test_x, test_y, kernel, optimizer_name, lr, max_epochs, 
                           device=None, use_multi_gpu=False, gpu_id=None, y_scaler=None):
    """Enhanced training function with multi-GPU support and better optimization"""
    
    if device is None:
        if gpu_id is not None and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Move data to device
        train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
        train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
        test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
        test_y = torch.tensor(test_y, dtype=torch.float32).to(device)

        # Initialize model and likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(train_x, train_y, likelihood, kernel.to(device)).to(device)

        # Note: DataParallel is not used for GPR training as it causes issues with 
        # GPyTorch's lazy tensors and MarginalLogLikelihood
        # Multi-GPU parallelism is achieved by running different folds on different GPUs
        # DO NOT wrap model or likelihood with DataParallel

        model.train()
        likelihood.train()

        # Enhanced optimizer setup
        optimizer_params = {'lr': lr}
        if optimizer_name == 'Adam':
            optimizer_params.update({'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-4})
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        elif optimizer_name == 'SGD':
            optimizer_params.update({'momentum': 0.9, 'weight_decay': 1e-4})
            optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer_name == 'Adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
        elif optimizer_name == 'AdamW':
            optimizer_params.update({'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-2})
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
        elif optimizer_name == 'Adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
        elif optimizer_name == 'LBFGS':
            optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, 
                                         tolerance_grad=1e-7, tolerance_change=1e-9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=10
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        losses = []
        best_loss = float('inf')
        patience_counter = 0

        pbar = tqdm(range(max_epochs), desc="Training", leave=False)
        
        for i in pbar:
            def closure():
                optimizer.zero_grad()
                with gpytorch.settings.cholesky_jitter(1e-1):
                    output = model(train_x)
                    loss = -mll(output, train_y)
                loss.backward()
                return loss

            if optimizer_name == 'LBFGS':
                loss = optimizer.step(closure)
                loss = loss.item() if hasattr(loss, 'item') else float(loss)
            else:
                optimizer.zero_grad()
                with gpytorch.settings.cholesky_jitter(1e-1):
                    output = model(train_x)
                    loss = -mll(output, train_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                loss = loss.item()

            losses.append(loss)
            pbar.set_postfix({'loss': f'{loss:.4f}', 'best': f'{best_loss:.4f}'})

            scheduler.step(loss)

            if loss < best_loss - early_stopping_delta:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.debug(f"Early stopping at epoch {i+1} (loss stabilized)")
                break
                
            if i % 10 == 0:
                gpu_manager.optimize_memory()

        pbar.close()

        # Enhanced evaluation
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(1e-1), gpytorch.settings.max_cg_iterations(1000):
            
            preds = likelihood(model(test_x))
            y_pred = preds.mean
            
            y_true_z = test_y.cpu().numpy().reshape(-1,1)
            y_pred_z = y_pred.cpu().numpy().reshape(-1,1)
            
            # Use provided y_scaler if available, otherwise compute metrics on scaled data
            if y_scaler is not None:
                y_true_m = y_scaler.inverse_transform(y_true_z).ravel()
                y_pred_m = y_scaler.inverse_transform(y_pred_z).ravel()
            else:
                y_true_m = y_true_z.ravel()
                y_pred_m = y_pred_z.ravel()
                
            rmse = np.sqrt(mean_squared_error(y_true_m, y_pred_m))
            r2 = r2_score(y_true_m, y_pred_m)
        
        # Clean up to prevent memory leaks and lazy tensor issues
        if gpu_id is not None:
            del train_x, train_y, test_x, test_y, preds, y_pred
            torch.cuda.empty_cache()
            
        return rmse, r2, model, likelihood, losses
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"GPU out of memory during training on device {device}")
            gpu_manager.optimize_memory()
            return float('inf'), -float('inf'), None, None, []
        else:
            logger.error(f"Runtime error during training: {e}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        return float('inf'), -float('inf'), None, None, []

def train_and_eval(train_x, train_y, test_x, test_y, kernel, optimizer_name, lr, max_epochs):
    """Legacy wrapper for backward compatibility"""
    return train_and_eval_enhanced(train_x, train_y, test_x, test_y, kernel, optimizer_name, lr, max_epochs)

################################################################################################################
# Parallel cross-validation with multi-GPU support
################################################################################################################

def parallel_cross_validation_fold(args):
    """
    Unified function to process a single cross-validation fold on GPU or CPU.
    Works for both GPU (with gpu_id) and CPU (gpu_id=None) modes.
    """
    # Unpack args - handle both GPU and CPU cases
    if len(args) == 11:  # GPU mode: includes gpu_id
        fold, train_idx, test_idx, X_subset, y_subset, kernel_serialized, opt, lr, epochs, gpu_id, y_scaler = args
        is_gpu = True
    else:  # CPU mode: no gpu_id
        fold, train_idx, test_idx, X_subset, y_subset, kernel_serialized, opt, lr, epochs, y_scaler = args
        gpu_id = None
        is_gpu = False
    
    try:
        # Small delay to prevent all workers starting simultaneously
        time.sleep(0.1 * (fold - 1) if is_gpu else 0.05 * (fold - 1))
        
        # Setup device
        if is_gpu and torch.cuda.is_available():
            # GPU mode: setup GPU
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(gpu_id)
            torch.cuda.reset_peak_memory_stats(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
            device_name = f"GPU {gpu_id}"
        else:
            # CPU mode
            device = torch.device('cpu')
            device_name = "CPU"
        
        # Log start of training
        print(f"[FOLD {fold}] Starting training on {device_name}...", flush=True)
        
        # Reconstruct kernel on the target device
        kernel = build_kernel_from_serialized_enhanced(kernel_serialized)
        
        # Train and evaluate (unified call works for both GPU and CPU)
        rmse, r2, _, _, _ = train_and_eval_enhanced(
            X_subset[train_idx], y_subset[train_idx], 
            X_subset[test_idx], y_subset[test_idx],
            kernel, opt, lr, epochs, 
            gpu_id=gpu_id if is_gpu else None,
            device=device if not is_gpu else None,
            y_scaler=y_scaler
        )
        
        print(f"[FOLD {fold}] Training completed on {device_name}: RMSE={rmse:.4f}, R²={r2:.4f}", flush=True)
        
        # Cleanup
        del kernel
        if is_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(gpu_id)
        gc.collect()
        
        return fold, rmse, r2
        
    except RuntimeError as e:
        if "out of memory" in str(e) or "CUDA" in str(e):
            print(f"[FOLD {fold}] Memory error on {device_name}: {e}", flush=True)
            logger.error(f"Memory error in fold {fold} on {device_name}: {e}")
            if is_gpu and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(gpu_id)
                except:
                    pass
            return fold, float('inf'), -float('inf')
        else:
            print(f"[FOLD {fold}] Runtime error on {device_name}: {e}", flush=True)
            logger.error(f"Runtime error in fold {fold} on {device_name}: {e}")
            return fold, float('inf'), -float('inf')
    except Exception as e:
        print(f"[FOLD {fold}] Error on {device_name}: {e}", flush=True)
        logger.error(f"Error in fold {fold} on {device_name}: {e}")
        if is_gpu and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
        gc.collect()
        return fold, float('inf'), -float('inf')

################################################################################################################
# Load and preprocess data
################################################################################################################

if __name__ == '__main__':
    # Load data with enhanced validation
    X_ftf, y_ftf = load_and_validate_data()
    X, y, X_scaler, y_scaler = preprocess_data(X_ftf, y_ftf)

    # Setup device and multi-GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ================================================
    # Unified Hardware Selection (GPU or CPU)
    # ================================================
    num_cpu_workers = 1  # Used for CPU-only mode
    
    if torch.cuda.is_available():
        # GPU Mode: Options 1-4
        total_gpus = torch.cuda.device_count()
        logger.info(f"\nDetected {total_gpus} GPU(s):")
        for i in range(total_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name}, Memory: {props.total_memory / 1e9:.1f} GB")
        
        # Check if CUDA_VISIBLE_DEVICES is already set
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_visible:
            logger.info(f"\nCUDA_VISIBLE_DEVICES is set to: {cuda_visible}")
            logger.info(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")
        else:
            # Ask user for GPU selection (Options 1-4)
            print("\n" + "="*70)
            print("Hardware Selection - GPU Mode:")
            print("="*70)
            print("1. Use ALL available GPUs (parallel training & prediction)")
            print("2. Use SINGLE GPU (sequential, no parallelism)")
            print("3. Select SPECIFIC GPUs (e.g., skip failing GPUs)")
            print("4. Use CPU instead (will ask for number of cores)")
            print("="*70)
            
            try:
                choice = input("\nEnter option (1/2/3/4) [default: 1]: ").strip()
                if not choice:
                    choice = '1'
                
                if choice == '1':
                    logger.info(f"Using ALL {total_gpus} GPUs")
                    
                elif choice == '2':
                    gpu_id = input(f"Enter GPU ID to use (0-{total_gpus-1}) [default: 0]: ").strip()
                    gpu_id = int(gpu_id) if gpu_id else 0
                    if 0 <= gpu_id < total_gpus:
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                        torch.cuda.empty_cache()
                        logger.info(f"Using SINGLE GPU {gpu_id} only")
                        logger.info(f"PyTorch now sees {torch.cuda.device_count()} GPU")
                    else:
                        logger.warning(f"Invalid GPU ID. Using GPU 0")
                        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                        
                elif choice == '3':
                    print(f"\nEnter GPU IDs to use (comma-separated, e.g., 1,2,5,6,7)")
                    print(f"Available GPUs: 0-{total_gpus-1}")
                    gpu_list = input("GPU IDs: ").strip()
                    if gpu_list:
                        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
                        torch.cuda.empty_cache()
                        logger.info(f"Using GPUs: {gpu_list}")
                        logger.info(f"PyTorch now sees {torch.cuda.device_count()} GPU(s)")
                    else:
                        logger.warning("No GPUs specified. Using all GPUs")
                
                elif choice == '4':
                    # User chose CPU mode - disable GPUs
                    logger.info("User selected CPU mode - disabling GPUs")
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
                    torch.cuda.empty_cache()
                    device = torch.device('cpu')
                    # Continue to CPU selection below
                    
                else:
                    logger.warning(f"Invalid choice '{choice}'. Using all GPUs")
                    
            except (ValueError, KeyboardInterrupt) as e:
                logger.warning(f"Input error: {e}. Using all available GPUs")
            
            print("="*70 + "\n")
    
    # CPU Mode: Ask for number of cores (same for training & prediction)
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        import multiprocessing as mp_cpu
        total_cpus = mp_cpu.cpu_count()
        logger.info(f"\nCPU Mode - System has {total_cpus} CPU cores available")
        
        print("\n" + "="*70)
        print("Hardware Selection - CPU Mode:")
        print("="*70)
        print(f"Enter number of CPU cores to use (1-{total_cpus})")
        print(f"  - Use 1 for sequential processing (slowest)")
        print(f"  - Use {total_cpus} for maximum parallelism (recommended)")
        print("  - This setting applies to BOTH training and prediction")
        print("="*70)
        
        try:
            cpu_input = input(f"\nNumber of CPU cores [default: {total_cpus}]: ").strip()
            if cpu_input:
                num_cpu_workers = int(cpu_input)
                num_cpu_workers = max(1, min(num_cpu_workers, total_cpus))
            else:
                num_cpu_workers = total_cpus
            logger.info(f"Using {num_cpu_workers} CPU core(s) for training and prediction")
        except (ValueError, KeyboardInterrupt) as e:
            logger.warning(f"Input error: {e}. Using all {total_cpus} CPU cores")
            num_cpu_workers = total_cpus
        
        print("="*70 + "\n")
    
    # Setup multi-GPU environment
    # Note: For GPR, we use multi-GPU by distributing CV folds across GPUs,
    # not by using DataParallel within training (which causes issues with GPyTorch)
    use_multi_gpu = gpu_manager.setup_multi_gpu(use_distributed=False)
    
    # Log final configuration
    final_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if final_gpu_count == 0:
        if num_cpu_workers == 1:
            logger.info("Running on SINGLE CPU core (sequential processing)")
        else:
            logger.info(f"Running on {num_cpu_workers} CPU cores (parallel processing)")
    elif final_gpu_count == 1:
        logger.info("Running in SINGLE-GPU mode (sequential processing)")
    else:
        logger.info(f"Running in MULTI-GPU mode with {final_gpu_count} GPUs (parallel processing)")

    ################################################################################################################
    # Define kernels and kernel parameters
    ################################################################################################################
    kernel_options = [
    # gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1])),
    # gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=X.shape[1])),
    # gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=X.shape[1])),
    # gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=X.shape[1])),
    # gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2, ard_num_dims=X.shape[1])),
    # gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=3, ard_num_dims=X.shape[1])),
    # gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(ard_num_dims=X.shape[1])),
    # gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(ard_num_dims=X.shape[1])),
    # gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2, ard_num_dims=X.shape[1])),
    # # Composite kernels: sum or product, then wrap in a single ScaleKernel
    # gpytorch.kernels.ScaleKernel(
    #     gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]) + gpytorch.kernels.LinearKernel(ard_num_dims=X.shape[1])
    # ),
    # gpytorch.kernels.ScaleKernel(
    #     gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]) * gpytorch.kernels.LinearKernel(ard_num_dims=X.shape[1])
    # ),
    # gpytorch.kernels.ScaleKernel(
    #     gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]) + gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=X.shape[1])
    # ),
    # gpytorch.kernels.ScaleKernel(
    #     gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]) * gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=X.shape[1])
    # ),
    # gpytorch.kernels.ScaleKernel(
    #     gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]) + gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=X.shape[1])
    # ),
    # gpytorch.kernels.ScaleKernel(
    #     gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]) * gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=X.shape[1])
    # ),
    gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]) + gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=X.shape[1])
    ),
    # gpytorch.kernels.ScaleKernel(
        # gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]) * gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=X.shape[1])
    # ),
  ]

    kernel_hyperparams = {
        # 'RBFKernel': {'lengthscale': [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]},
        'RBFKernel': {'lengthscale': [1.0]},
        # 'MaternKernel': {'nu': [0.5, 1.5, 2.5], 'lengthscale': [0.5, 1.0, 2.0]},
        'MaternKernel': {'nu': [2.5], 'lengthscale': [1.0]},
        'PolynomialKernel': {'power': [2, 3], 'offset': [0.0, 1.0, 2.0, 5.0]},  # Try a wider range
        'LinearKernel': {},
        'PeriodicKernel': {'period_length': [1.0, 2.0]},
        'SpectralMixtureKernel': {'num_mixtures': [2], 'ard_num_dims': [X.shape[1]]},
    }

def extract_kernel_types(kernel):
    """Recursively extract all base kernel types from a (possibly composite) kernel."""
    if isinstance(kernel, gpytorch.kernels.ScaleKernel):
        return extract_kernel_types(kernel.base_kernel)
    elif isinstance(kernel, (gpytorch.kernels.AdditiveKernel, gpytorch.kernels.ProductKernel)):
        types = []
        for i, k in enumerate(kernel.kernels):
            for subpath, subtype in extract_kernel_types(k):
                types.append(((i,) + subpath, subtype))
        return types
    else:
        return [((), type(kernel).__name__)]

def build_kernel_with_params(kernel, param_dict, path=()):
    if isinstance(kernel, gpytorch.kernels.ScaleKernel):
        base = build_kernel_with_params(kernel.base_kernel, param_dict, path)
        return gpytorch.kernels.ScaleKernel(base)
    elif isinstance(kernel, gpytorch.kernels.AdditiveKernel):
        subkernels = [build_kernel_with_params(k, param_dict, path + (i,)) for i, k in enumerate(kernel.kernels)]
        return gpytorch.kernels.AdditiveKernel(*subkernels)
    elif isinstance(kernel, gpytorch.kernels.ProductKernel):
        subkernels = [build_kernel_with_params(k, param_dict, path + (i,)) for i, k in enumerate(kernel.kernels)]
        return gpytorch.kernels.ProductKernel(*subkernels)
    else:
        ktype = type(kernel).__name__
        params = param_dict.get(path, {})
        if ktype == 'RBFKernel':
            return gpytorch.kernels.RBFKernel(lengthscale=params.get('lengthscale', 1.0), ard_num_dims=X.shape[1])
        elif ktype == 'MaternKernel':
            return gpytorch.kernels.MaternKernel(nu=params.get('nu', 2.5), lengthscale=params.get('lengthscale', 1.0), ard_num_dims=X.shape[1])
        elif ktype == 'PolynomialKernel':
            return gpytorch.kernels.PolynomialKernel(power=params.get('power', 2), offset=params.get('offset', 0.0), ard_num_dims=X.shape[1])
        elif ktype == 'SpectralMixtureKernel':
            return gpytorch.kernels.SpectralMixtureKernel(
                num_mixtures=params.get('num_mixtures', 2),
                ard_num_dims=params.get('ard_num_dims', X.shape[1])
            )
        elif ktype == 'LinearKernel':
            return gpytorch.kernels.LinearKernel(ard_num_dims=X.shape[1])
        elif ktype == 'PeriodicKernel':
            return gpytorch.kernels.PeriodicKernel(period_length=params.get('period_length', 1.0), ard_num_dims=X.shape[1])
        else:
            raise ValueError(f"Unknown kernel type: {ktype}")

################################################################################################################
# Define optimizers
################################################################################################################
optimizer_options = [
    'Adam',
    # 'SGD',
    # 'RMSprop',
    # 'Adagrad',
    # 'AdamW',
    # 'Adadelta',
    # 'LBFGS'
]
optimizer_hyperparams = {
    'Adam': {'lr': learning_rate_options, 'weight_decay': [0, 1e-4]},
    'SGD': {'lr': learning_rate_options, 'momentum': [0.0, 0.9]},
    'RMSprop': {'lr': learning_rate_options},
    'Adagrad': {'lr': learning_rate_options},
    'AdamW': {'lr': learning_rate_options, 'weight_decay': [0, 1e-4]},
    'Adadelta': {'lr': learning_rate_options},
    'LBFGS': {'lr': learning_rate_options}
}

################################################################################################################
# Enhanced kernel serialization functions
################################################################################################################
def serialize_kernel(kernel):
    """Enhanced kernel serialization with better error handling"""
    try:
        if isinstance(kernel, gpytorch.kernels.ScaleKernel):
            return {
                'kernel_type': 'ScaleKernel',
                'base_kernel': serialize_kernel(kernel.base_kernel)
            }
        elif isinstance(kernel, gpytorch.kernels.AdditiveKernel):
            return {
                'kernel_type': 'AdditiveKernel',
                'subkernels': [serialize_kernel(k) for k in kernel.kernels]
            }
        elif isinstance(kernel, gpytorch.kernels.ProductKernel):
            return {
                'kernel_type': 'ProductKernel',
                'subkernels': [serialize_kernel(k) for k in kernel.kernels]
            }
        elif isinstance(kernel, gpytorch.kernels.RBFKernel):
            return {
                'kernel_type': 'RBFKernel',
                'params': {
                    'lengthscale': kernel.lengthscale.detach().cpu().numpy().tolist(),
                    'ard_num_dims': kernel.ard_num_dims if hasattr(kernel, 'ard_num_dims') else None
                }
            }
        elif isinstance(kernel, gpytorch.kernels.MaternKernel):
            return {
                'kernel_type': 'MaternKernel',
                'params': {
                    'nu': float(kernel.nu) if hasattr(kernel, 'nu') else 2.5,
                    'lengthscale': kernel.lengthscale.detach().cpu().numpy().tolist(),
                    'ard_num_dims': kernel.ard_num_dims if hasattr(kernel, 'ard_num_dims') else None
                }
            }
        elif isinstance(kernel, gpytorch.kernels.PolynomialKernel):
            return {
                'kernel_type': 'PolynomialKernel',
                'params': {
                    'power': int(kernel.power) if hasattr(kernel, 'power') else 2,
                    'offset': float(kernel.offset) if hasattr(kernel, 'offset') else 0.0,
                    'lengthscale': kernel.lengthscale.detach().cpu().numpy().tolist() if hasattr(kernel, 'lengthscale') else None,
                    'ard_num_dims': kernel.ard_num_dims if hasattr(kernel, 'ard_num_dims') else None
                }
            }
        elif isinstance(kernel, gpytorch.kernels.LinearKernel):
            return {
                'kernel_type': 'LinearKernel',
                'params': {
                    'variance': kernel.variance.detach().cpu().numpy().tolist() if hasattr(kernel, 'variance') else None,
                    'ard_num_dims': kernel.ard_num_dims if hasattr(kernel, 'ard_num_dims') else None
                }
            }
        elif isinstance(kernel, gpytorch.kernels.PeriodicKernel):
            return {
                'kernel_type': 'PeriodicKernel',
                'params': {
                    'period_length': kernel.period_length.detach().cpu().numpy().tolist() if hasattr(kernel, 'period_length') else 1.0,
                    'lengthscale': kernel.lengthscale.detach().cpu().numpy().tolist() if hasattr(kernel, 'lengthscale') else None,
                    'ard_num_dims': kernel.ard_num_dims if hasattr(kernel, 'ard_num_dims') else None
                }
            }
        elif isinstance(kernel, gpytorch.kernels.SpectralMixtureKernel):
            return {
                'kernel_type': 'SpectralMixtureKernel',
                'params': {
                    'num_mixtures': kernel.num_mixtures if hasattr(kernel, 'num_mixtures') else 2,
                    'ard_num_dims': kernel.ard_num_dims if hasattr(kernel, 'ard_num_dims') else None
                }
            }
        else:
            logger.warning(f"Unknown kernel type for serialization: {type(kernel)}")
            raise ValueError(f"Unknown kernel type: {type(kernel)}")
    except Exception as e:
        logger.error(f"Error serializing kernel: {e}")
        raise

def build_kernel_from_serialized_enhanced(kernel_dict):
    """Enhanced kernel reconstruction with better error handling"""
    kernel_type = kernel_dict['kernel_type']
    if kernel_type == 'ScaleKernel':
        return gpytorch.kernels.ScaleKernel(build_kernel_from_serialized_enhanced(kernel_dict['base_kernel']))
    elif kernel_type == 'AdditiveKernel':
        subkernels = [build_kernel_from_serialized_enhanced(sub) for sub in kernel_dict['subkernels']]
        return gpytorch.kernels.AdditiveKernel(*subkernels)
    elif kernel_type == 'ProductKernel':
        subkernels = [build_kernel_from_serialized_enhanced(sub) for sub in kernel_dict['subkernels']]
        return gpytorch.kernels.ProductKernel(*subkernels)
    elif kernel_type == 'RBFKernel':
        ard_num_dims = kernel_dict['params'].get('ard_num_dims')
        return gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
    elif kernel_type == 'MaternKernel':
        ard_num_dims = kernel_dict['params'].get('ard_num_dims')
        return gpytorch.kernels.MaternKernel(
            nu=kernel_dict['params']['nu'], 
            ard_num_dims=ard_num_dims
        )
    elif kernel_type == 'PolynomialKernel':
        ard_num_dims = kernel_dict['params'].get('ard_num_dims')
        return gpytorch.kernels.PolynomialKernel(
            power=kernel_dict['params']['power'],
            offset=kernel_dict['params'].get('offset', 0.0),
            ard_num_dims=ard_num_dims
        )
    elif kernel_type == 'LinearKernel':
        ard_num_dims = kernel_dict['params'].get('ard_num_dims')
        return gpytorch.kernels.LinearKernel(ard_num_dims=ard_num_dims)
    elif kernel_type == 'PeriodicKernel':
        ard_num_dims = kernel_dict['params'].get('ard_num_dims')
        return gpytorch.kernels.PeriodicKernel(
            period_length=kernel_dict['params']['period_length'],
            ard_num_dims=ard_num_dims
        )
    elif kernel_type == 'SpectralMixtureKernel':
        return gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=kernel_dict['params']['num_mixtures'],
            ard_num_dims=kernel_dict['params'].get('ard_num_dims')
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

# The main training loop continues from the if __name__ block started at line 493
# All variables (X, y, kernel_options, etc.) are still in scope here
if __name__ == '__main__':  
    ################################################################################################################
    # Enhanced main training loop with multi-GPU support
    ################################################################################################################

    # sample_sizes = [50, 150, 250, 350, 450, 550, 650, 750]
    sample_sizes = [750]
    rmse_folds_by_size = []
    r2_folds_by_size = []
    performance_metrics = []

    logger.info(f"Starting enhanced multi-GPU training with sample sizes: {sample_sizes}")

    memory_info = gpu_manager.get_memory_info()
    if memory_info:
        logger.info("GPU Memory Information:")
        for info in memory_info:
            logger.info(f"  GPU {info['gpu_id']}: {info['free_gb']:.1f}GB free / {info['total_gb']:.1f}GB total")

    for size_idx, size in enumerate(sample_sizes):
        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-validation and grid search for Sample Size: {size}")
        logger.info(f"{'='*60}")
    
        X_subset = X[:size]
        y_subset = y[:size]
    
        kf = KFold(n_splits=5, shuffle=True, random_state=seed_id)
        results = []
        start_time = time.time()
    
        # Progress tracking
        total_combinations = 0
        for kernel_option in kernel_options:
            subkernel_types = extract_kernel_types(kernel_option)
            subkernel_param_grids = []
            for path, ktype in subkernel_types:
                grid = kernel_hyperparams.get(ktype, {})
                if grid:
                    keys, values = zip(*grid.items())
                    combos = [dict(zip(keys, v)) for v in product(*values)]
                else:
                    combos = [{}]
                subkernel_param_grids.append(combos)
        
            for subkernel_param_combo in product(*subkernel_param_grids):
                for opt in optimizer_options:
                    opt_param_grid = optimizer_hyperparams.get(opt, {})
                    if opt_param_grid:
                        okeys, ovalues = zip(*opt_param_grid.items())
                        opt_param_combos = [dict(zip(okeys, v)) for v in product(*ovalues)]
                    else:
                        opt_param_combos = [{}]
                    for opt_params in opt_param_combos:
                        for lr in [opt_params.get('lr', 0.01)]:
                            for epochs in epoch_options:
                                total_combinations += 1
    
        logger.info(f"Total combinations to test: {total_combinations}")
        combination_count = 0
    
        for kernel_option in kernel_options:
            subkernel_types = extract_kernel_types(kernel_option)
            subkernel_param_grids = []
            for path, ktype in subkernel_types:
                grid = kernel_hyperparams.get(ktype, {})
                if grid:
                    keys, values = zip(*grid.items())
                    combos = [dict(zip(keys, v)) for v in product(*values)]
                else:
                    combos = [{}]
                subkernel_param_grids.append(combos)
        
            for subkernel_param_combo in product(*subkernel_param_grids):
                param_dict = {path: params for (path, _), params in zip(subkernel_types, subkernel_param_combo)}
                kernel = build_kernel_with_params(kernel_option, param_dict)
                for opt in optimizer_options:
                    opt_param_grid = optimizer_hyperparams.get(opt, {})
                    if opt_param_grid:
                        okeys, ovalues = zip(*opt_param_grid.items())
                        opt_param_combos = [dict(zip(okeys, v)) for v in product(*ovalues)]
                    else:
                        opt_param_combos = [{}]
                    for opt_params in opt_param_combos:
                        for lr in [opt_params.get('lr', 0.01)]:
                            for epochs in epoch_options:
                                combination_count += 1
                                logger.info(f"Testing combination {combination_count}/{total_combinations}")
                            
                                rmses, r2s = [], []
                            
                                # Enhanced parallel cross-validation across GPUs or CPUs
                                # Use ProcessPoolExecutor instead of ThreadPoolExecutor to avoid
                                # GPyTorch lazy tensor thread-safety issues
                                
                                # Determine if we should use parallel processing
                                use_parallel = False
                                max_workers = 1
                                
                                if use_multi_gpu and torch.cuda.device_count() > 1:
                                    # Multi-GPU parallel processing
                                    use_parallel = True
                                    max_workers = min(torch.cuda.device_count(), 5)
                                    worker_type = 'GPU'
                                elif num_cpu_workers > 1:
                                    # Multi-CPU parallel processing
                                    use_parallel = True
                                    max_workers = min(num_cpu_workers, 5)  # Limit to 5 concurrent CV folds
                                    worker_type = 'CPU'
                                
                                if use_parallel:
                                    # Serialize kernel to avoid device conflicts when sharing across processes
                                    kernel_serialized = serialize_kernel(kernel)
                                
                                    fold_args = []
                                    for fold, (train_idx, test_idx) in enumerate(kf.split(X_subset), 1):
                                        if worker_type == 'GPU':
                                            gpu_id = (fold - 1) % torch.cuda.device_count()
                                            fold_args.append((fold, train_idx, test_idx, X_subset, y_subset, 
                                                            kernel_serialized, opt, lr, epochs, gpu_id, y_scaler))
                                        else:  # CPU
                                            fold_args.append((fold, train_idx, test_idx, X_subset, y_subset, 
                                                            kernel_serialized, opt, lr, epochs, y_scaler))
                                
                                    # Use ProcessPoolExecutor for true parallel execution
                                    # Each process has its own Python interpreter and memory space
                                    try:
                                        logger.info(f"Submitting {len(fold_args)} folds to {max_workers} parallel {worker_type} workers...")
                                        start_parallel = time.time()
                                        
                                        with ProcessPoolExecutor(max_workers=max_workers) as executor:
                                            # Submit all folds concurrently (unified function works for both GPU and CPU)
                                            futures = [executor.submit(parallel_cross_validation_fold, args) for args in fold_args]
                                            
                                            logger.info(f"All {len(futures)} folds submitted. Training in parallel on {worker_type}...")
                                            
                                            # Wait for all to complete and collect results
                                            for i, future in enumerate(futures, 1):
                                                fold, rmse, r2 = future.result()
                                                rmses.append(rmse)
                                                r2s.append(r2)
                                                logger.info(f"Fold {fold} completed ({i}/{len(futures)}): RMSE: {rmse:.4f} m, R²: {r2:.4f}")
                                        
                                        elapsed_parallel = time.time() - start_parallel
                                        logger.info(f"Parallel CV on {worker_type} completed in {elapsed_parallel:.1f}s (avg {elapsed_parallel/len(fold_args):.1f}s per fold)")
                                    except Exception as e:
                                        logger.error(f"ProcessPoolExecutor failed: {e}")
                                        logger.info("Falling back to sequential execution...")
                                        use_parallel = False  # Force sequential fallback
                                
                                if not use_parallel:
                                    # Sequential processing on single GPU/CPU
                                    logger.info("Running sequential cross-validation...")
                                    for fold, (train_idx, test_idx) in enumerate(kf.split(X_subset), 1):
                                        rmse, r2, _, _, _ = train_and_eval_enhanced(
                                            X_subset[train_idx], y_subset[train_idx], 
                                            X_subset[test_idx], y_subset[test_idx],
                                            kernel, opt, lr, epochs, device=device, use_multi_gpu=False, y_scaler=y_scaler
                                        )
                                        rmses.append(rmse)
                                        r2s.append(r2)
                                        logger.info(f"Fold {fold} RMSE: {rmse:.4f} m, R²: {r2:.4f}")
                            
                                avg_rmse = np.mean(rmses)
                                avg_r2 = np.mean(r2s)
                            
                                results.append({
                                    'kernel': kernel,
                                    'kernel_option': kernel_option,
                                    'kernel_param_dict': param_dict,
                                    'optimizer': opt,
                                    'optimizer_params': opt_params,
                                    'rmse': avg_rmse,
                                    'r2': avg_r2,
                                    'rmse_folds': rmses,
                                    'r2_folds': r2s,
                                    'learning_rate': lr,
                                    'epochs': epochs
                                })
                            
                                gpu_manager.optimize_memory()
                                gc.collect()

        # Find best model
        valid_results = [r for r in results if r['rmse'] != float('inf')]
        if not valid_results:
            logger.error(f"No valid results for sample size {size}")
            continue
        
        best = min(valid_results, key=lambda x: x['rmse'])
        logger.info(f"\nBest combination for sample size {size}:")
        logger.info(f"  RMSE: {best['rmse']:.4f} ± {np.std(best['rmse_folds']):.4f}")
        logger.info(f"  R²: {best['r2']:.4f} ± {np.std(best['r2_folds']):.4f}")
        logger.info(f"  Optimizer: {best['optimizer']} (lr={best['learning_rate']}, epochs={best['epochs']})")
    
        rmse_folds_by_size.append(best['rmse_folds'])
        r2_folds_by_size.append(best['r2_folds'])
    
        performance_metrics.append({
            'sample_size': size,
            'best_rmse': best['rmse'],
            'best_r2': best['r2'],
            'rmse_std': np.std(best['rmse_folds']),
            'r2_std': np.std(best['r2_folds']),
            'optimizer': best['optimizer'],
            'learning_rate': best['learning_rate'],
            'epochs': best['epochs']
        })

        # Only save the model for the largest sample size
        if size_idx == len(sample_sizes) - 1:
            logger.info("Training final model on full dataset...")
        
            best_kernel = build_kernel_with_params(best['kernel_option'], best['kernel_param_dict'])
            best_lr = best['learning_rate']
            best_epochs = best['epochs']
            best_opt = best['optimizer']
        
            final_rmse, final_r2, final_model, final_likelihood, final_losses = train_and_eval_enhanced(
                X_subset, y_subset, X_subset, y_subset, best_kernel, best_opt, best_lr, best_epochs,
                device=device, use_multi_gpu=use_multi_gpu
            )
        
            # Enhanced model saving
            kernel_serialized = serialize_kernel(best_kernel)
            model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_gpytorch_model_enhanced.pth")
        
            model_state_dict = final_model.module.state_dict() if hasattr(final_model, 'module') else final_model.state_dict()
            likelihood_state_dict = final_likelihood.module.state_dict() if hasattr(final_likelihood, 'module') else final_likelihood.state_dict()
        
            torch.save({
                'model_state_dict': model_state_dict,
                'likelihood_state_dict': likelihood_state_dict,
                'kernel_serialized': kernel_serialized,
                'kernel_param_dict': best['kernel_param_dict'],
                'optimizer': best_opt,
                'optimizer_params': best['optimizer_params'],
                'kernel_option_str': str(best['kernel_option']),
                'training_info': {
                    'sample_size': size,
                    'final_rmse': final_rmse,
                    'final_r2': final_r2,
                    'gpu_count': torch.cuda.device_count(),
                    'use_multi_gpu': use_multi_gpu
                }
            }, model_save_path)
        
            joblib.dump(X_scaler, os.path.join(os.path.dirname(os.path.abspath(__file__)), "X_scaler_enhanced.save"))
            joblib.dump(y_scaler, os.path.join(os.path.dirname(os.path.abspath(__file__)), "y_scaler_enhanced.save"))
            logger.info(f"Enhanced model saved to {model_save_path}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nSample size {size} finished --- {elapsed_time:.2f} seconds ---\n")
        logger.info(f"Sample size {size} completed in {elapsed_time:.2f} seconds")
    
        if torch.cuda.is_available():
            current_memory = gpu_manager.get_memory_info()
            logger.info("Current GPU memory usage:")
            for info in current_memory:
                logger.info(f"  GPU {info['gpu_id']}: {info['allocated_gb']:.1f}GB allocated, {info['free_gb']:.1f}GB free")

    # Enhanced results saving
    results_df = pd.DataFrame({
        'SampleSize': sample_sizes[:len(rmse_folds_by_size)], 
        'RMSE': rmse_folds_by_size, 
        'R2': r2_folds_by_size
    })
    results_df.to_csv("GPR_Enhanced_RMSE_R2_by_SampleSize.csv", index=False)

    performance_df = pd.DataFrame(performance_metrics)
    performance_df.to_csv("GPR_Enhanced_Performance_Metrics.csv", index=False)

    logger.info("Enhanced RMSE and R² results saved to CSV files.")

    # Enhanced plotting - only plot if we have valid results
    if len(rmse_folds_by_size) > 0:
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.boxplot(rmse_folds_by_size, tick_labels=sample_sizes[:len(rmse_folds_by_size)])
        plt.xlabel("Sample Size", fontsize=12)
        plt.ylabel("RMSE (m)", fontsize=12)
        plt.title("RMSE across Sample Sizes", fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 2)
        plt.boxplot(r2_folds_by_size, tick_labels=sample_sizes[:len(r2_folds_by_size)])
        plt.xlabel("Sample Size", fontsize=12)
        plt.ylabel("R²", fontsize=12)
        plt.title("R² across Sample Sizes", fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 3)
        mean_rmse = [np.mean(rmse) for rmse in rmse_folds_by_size]
        std_rmse = [np.std(rmse) for rmse in rmse_folds_by_size]
        sizes = sample_sizes[:len(rmse_folds_by_size)]

        plt.errorbar(sizes, mean_rmse, yerr=std_rmse, marker='o', capsize=5)
        plt.xlabel("Sample Size", fontsize=12)
        plt.ylabel("Mean RMSE (m)", fontsize=12)
        plt.title("RMSE Convergence", fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 4)
        mean_r2 = [np.mean(r2) for r2 in r2_folds_by_size]
        std_r2 = [np.std(r2) for r2 in r2_folds_by_size]

        plt.errorbar(sizes, mean_r2, yerr=std_r2, marker='s', capsize=5, color='orange')
        plt.xlabel("Sample Size", fontsize=12)
        plt.ylabel("Mean R²", fontsize=12)
        plt.title("R² Convergence", fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 5)
        if 'final_losses' in locals() and final_losses:
            plt.plot(final_losses)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Negative Log Likelihood", fontsize=12)
            plt.title("Final Training Loss", fontsize=14)
            plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 6)
        info_text = f"GPUs: {torch.cuda.device_count()}\n"
        info_text += f"Multi-GPU: {use_multi_gpu}\n"
        info_text += f"Device: {device}\n"
        if torch.cuda.is_available():
            info_text += f"Max Memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB"

        plt.text(0.1, 0.5, info_text, fontsize=11, transform=plt.gca().transAxes, 
                 verticalalignment='center')
        plt.title("System Information", fontsize=14)
        plt.axis('off')

        plt.suptitle("Enhanced GPR Model Performance Metrics", fontsize=16)
        plt.tight_layout()

        # Use platform-independent path for HPC compatibility
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fig_dir = os.path.join(script_dir, "Fig")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, "GPR_Enhanced_RMSE_R2_by_SampleSize.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
        # Use non-interactive backend for HPC - don't call plt.show() if no display
        try:
            plt.show()
        except Exception as e:
            logger.warning(f"Could not display plot (no display available): {e}")
            plt.close()
    
        logger.info(f"Enhanced performance plots saved to {fig_path}")
    else:
        logger.warning("No valid results to plot. Skipping visualization.")

    ################################################################################################################
    # Enhanced prediction with multi-GPU support
    ################################################################################################################

# Module-level worker function for parallel GPU prediction (must be at module level for pickling)
def _predict_worker(worker_id, start_idx, end_idx, pred_x_tensor_np, batch_size, 
                   model_state_dict, likelihood_state_dict, kernel_serialized, is_gpu=True):
    """
    Unified worker function to run prediction on GPU or CPU in parallel.
    
    Args:
        worker_id: GPU ID (if is_gpu=True) or CPU worker ID (if is_gpu=False)
        is_gpu: If True, use GPU; if False, use CPU
    """
    try:
        # Setup device
        if is_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{worker_id}')
            device_name = f"GPU {worker_id}"
        else:
            device = torch.device('cpu')
            device_name = f"CPU worker {worker_id}"
        
        # Reconstruct kernel on this device
        reconstructed_kernel = build_kernel_from_serialized_enhanced(kernel_serialized)
        
        # Convert numpy back to tensor
        pred_x_tensor = torch.from_numpy(pred_x_tensor_np).float()
        
        # Create dummy training data for model initialization
        dummy_x = torch.randn(10, pred_x_tensor.shape[1]).to(device)
        dummy_y = torch.randn(10).to(device)
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(dummy_x, dummy_y, likelihood, 
                           reconstructed_kernel.to(device), set_init=False).to(device)
        
        # Load state dict
        model.load_state_dict(model_state_dict)
        likelihood.load_state_dict(likelihood_state_dict)
        
        model.eval()
        likelihood.eval()
        
        # Get this worker's chunk of data
        batch_data = pred_x_tensor[start_idx:end_idx].to(device)
        batch_predictions = []
        
        # Process in smaller batches to avoid memory issues
        for i in range(0, batch_data.shape[0], batch_size):
            batch_end = min(i + batch_size, batch_data.shape[0])
            mini_batch = batch_data[i:batch_end]
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.cholesky_jitter(1e-1):
                preds = likelihood(model(mini_batch))
                y_pred_batch = preds.mean.cpu().numpy() if is_gpu else preds.mean.numpy()
                batch_predictions.append(y_pred_batch)
        
        result = np.concatenate(batch_predictions)
        
        # Cleanup
        del model, likelihood, batch_data
        if is_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            gc.collect()
        
        return result
        
    except Exception as e:
        logger.error(f"{device_name} prediction failed: {e}")
        raise

def predict_with_enhanced_model_multi_gpu(input_file='Inputs_GPR_predict_500yr_sim_1.csv', 
                                         model_path=None, output_file=None, batch_size=1000, num_workers=None):
    """Enhanced prediction function with multi-GPU/CPU support and batch processing"""
    start_time = time.time()
    
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_gpytorch_model_enhanced.pth")

    # ================================================
    # Use the same number of workers as training (passed as parameter)
    # ================================================
    num_available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if num_workers is None:
        # Fallback: if not provided, use GPU count or 1
        num_prediction_workers = num_available_gpus if num_available_gpus > 0 else 1
    else:
        # Use the same worker count as training
        num_prediction_workers = num_workers
    
    logger.info(f"Using {num_prediction_workers} worker(s) for prediction (same as training)")

    try:
        logger.info("Loading enhanced model for prediction...")
        
        # PyTorch 2.6+ requires weights_only=False for loading models with custom objects
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        logger.info(f"Kernel option (string): {checkpoint.get('kernel_option_str', 'Not saved')}")
        
        # Load scalers
        try:
            X_scaler_enhanced = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "X_scaler_enhanced.save"))
            y_scaler_enhanced = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "y_scaler_enhanced.save"))
        except FileNotFoundError:
            logger.warning("Enhanced scalers not found, falling back to regular scalers")
            X_scaler_enhanced = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "X_scaler.save"))
            y_scaler_enhanced = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "y_scaler.save"))

        # Get kernel serialization info (will be used for reconstruction on each GPU)
        kernel_serialized = checkpoint['kernel_serialized']
        logger.info(f"Kernel serialized successfully")

        # Load prediction data
        logger.info(f"Attempting to load prediction data from: {input_file}")
        input_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_file) if not os.path.isabs(input_file) else input_file
        
        # Normalize path for Windows (handles OneDrive paths better)
        input_file_path = os.path.normpath(input_file_path)
        
        if os.path.exists(input_file_path):
            file_size_mb = os.path.getsize(input_file_path) / (1024 * 1024)
            logger.info(f"Found input file: {input_file_path} ({file_size_mb:.2f} MB)")
        else:
            logger.info(f"Input file not found at: {input_file_path}")
        
        try:
            logger.info("Starting CSV read...")
            from pathlib import Path
            import platform
            
            file_path_obj = Path(input_file_path)
            
            # Choose engine based on platform for optimal performance
            # C engine: Fast but can have issues on Windows with OneDrive
            # Python engine: Slower but more robust on Windows
            is_windows = platform.system() == 'Windows'
            
            if is_windows:
                # Windows/CPU: use Python engine for OneDrive compatibility
                logger.info("Windows detected, using Python engine for compatibility...")
                pred_x_df = pd.read_csv(file_path_obj, header=None, engine='python')
            else:
                # Linux/GPU: use fast C engine for large files
                logger.info(f"{platform.system()} detected, using C engine for speed...")
                pred_x_df = pd.read_csv(file_path_obj, header=None, engine='c', low_memory=False)
            
            logger.info(f"Successfully loaded prediction data: {pred_x_df.shape}")
        except FileNotFoundError as e:
            logger.warning(f"Input file not found: {e}. Creating synthetic data...")
            n_samples = 8760  # 1 year hourly data
            n_features = X_scaler_enhanced.n_features_in_
            pred_x_df = pd.DataFrame(np.random.randn(n_samples, n_features))
            logger.info(f"Created synthetic prediction data: {pred_x_df.shape}")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            logger.warning(f"Attempting to read with chunksize...")
            try:
                # Try reading in chunks for very large files
                from pathlib import Path
                import platform
                file_path_obj = Path(input_file_path)
                is_windows = platform.system() == 'Windows'
                chunk_list = []
                chunk_size = 100000  # Read 100k rows at a time
                logger.info(f"Reading file in chunks of {chunk_size} rows...")
                
                # Use appropriate engine for chunked reading
                if is_windows:
                    # Windows: Python engine (no low_memory parameter)
                    for i, chunk in enumerate(pd.read_csv(file_path_obj, header=None, chunksize=chunk_size, engine='python')):
                        chunk_list.append(chunk)
                        if (i + 1) % 10 == 0:
                            logger.info(f"Read {(i+1) * chunk_size} rows...")
                else:
                    # Linux/GPU: C engine with low_memory for speed
                    for i, chunk in enumerate(pd.read_csv(file_path_obj, header=None, chunksize=chunk_size, engine='c', low_memory=False)):
                        chunk_list.append(chunk)
                        if (i + 1) % 10 == 0:
                            logger.info(f"Read {(i+1) * chunk_size} rows...")
                
                pred_x_df = pd.concat(chunk_list, ignore_index=True)
                logger.info(f"Successfully loaded prediction data via chunking: {pred_x_df.shape}")
            except Exception as e2:
                logger.error(f"Chunked reading also failed: {e2}")
                logger.warning("Creating synthetic data as fallback...")
                n_samples = 8760  # 1 year hourly data
                n_features = X_scaler_enhanced.n_features_in_
                pred_x_df = pd.DataFrame(np.random.randn(n_samples, n_features))
                logger.info(f"Created synthetic prediction data: {pred_x_df.shape}")

        # Preprocess prediction data
        logger.info("Preprocessing prediction data...")
        pred_x_np = pred_x_df.values
        degree_columns = [i for i in range(min(10, pred_x_np.shape[1]))]
        pred_x_np[:, degree_columns] = np.deg2rad(pred_x_np[:, degree_columns])
        pred_x_scaled = X_scaler_enhanced.transform(pred_x_np)
        logger.info("Creating prediction tensor...")
        pred_x_tensor = torch.tensor(pred_x_scaled, dtype=torch.float32)

        logger.info(f"Prediction data shape: {pred_x_tensor.shape}")
        
        # CRITICAL: Adjust batch size based on data size to prevent OOM
        # For very large datasets (>1M samples), use smaller batches
        total_samples = pred_x_tensor.shape[0]
        if total_samples > 1000000:
            batch_size = 100  # Very small batches for huge datasets
            logger.warning(f"Large dataset detected ({total_samples} samples). Using batch_size={batch_size}")
        elif total_samples > 100000:
            batch_size = 500
            logger.info(f"Using batch_size={batch_size} for {total_samples} samples")
        else:
            batch_size = 1000

        # Multi-GPU prediction setup
        # CRITICAL: Check both CUDA availability AND that device_count > 1
        num_available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        y_pred_scaled = None  # Initialize to track if multi-GPU succeeded
        
        logger.info(f"Prediction routing: num_available_gpus={num_available_gpus}, num_prediction_workers={num_prediction_workers}")
        
        if num_available_gpus > 1:
            logger.info(f"Using {num_available_gpus} GPUs for prediction")
            
            # CRITICAL: Clear GPU memory before prediction
            for gpu_id in range(num_available_gpus):
                try:
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(f"Could not clear GPU {gpu_id}: {e}")
            
            # Distribute prediction batches across GPUs
            num_gpus = num_available_gpus
            total_samples = pred_x_tensor.shape[0]
            samples_per_gpu = total_samples // num_gpus
            
            # Convert tensor to numpy for pickling (ProcessPoolExecutor requirement)
            pred_x_tensor_np = pred_x_tensor.numpy()
            
            # Get model/likelihood state dicts for passing to workers
            model_state_dict = checkpoint['model_state_dict']
            likelihood_state_dict = checkpoint['likelihood_state_dict']
            
            predictions = []
            
            # PARALLEL multi-GPU prediction using ProcessPoolExecutor
            # Each GPU runs in its own process with independent memory
            logger.info(f"Starting PARALLEL prediction on {num_gpus} GPUs using ProcessPoolExecutor")
            
            try:
                with ProcessPoolExecutor(max_workers=num_gpus, mp_context=mp.get_context('spawn')) as executor:
                    futures = []
                    for gpu_id in range(num_gpus):
                        start_idx = gpu_id * samples_per_gpu
                        end_idx = (gpu_id + 1) * samples_per_gpu if gpu_id < num_gpus - 1 else total_samples
                        
                        if start_idx < total_samples:
                            logger.info(f"Submitting GPU {gpu_id}: samples {start_idx}-{end_idx} ({end_idx-start_idx} samples)")
                            future = executor.submit(_predict_worker, gpu_id, start_idx, end_idx, 
                                                   pred_x_tensor_np, batch_size, model_state_dict, 
                                                   likelihood_state_dict, kernel_serialized, is_gpu=True)
                            futures.append((gpu_id, future))
                    
                    # Collect results in order
                    for gpu_id, future in futures:
                        try:
                            result = future.result()
                            predictions.append(result)
                            logger.info(f"GPU {gpu_id}: Completed {len(result)} predictions")
                        except Exception as e:
                            logger.error(f"GPU {gpu_id}: Prediction worker failed with error: {e}")
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            raise
                
                # Combine predictions
                y_pred_scaled = np.concatenate(predictions)
                logger.info(f"PARALLEL multi-GPU prediction completed on {num_gpus} GPUs")
            except Exception as e:
                logger.error(f"ProcessPoolExecutor failed: {e}")
                logger.warning("Falling back to single-device prediction")
                y_pred_scaled = None  # Mark as failed, will use fallback
        
        # Single GPU/CPU prediction (fallback or primary path)
        if y_pred_scaled is None:
            # Check if we should use multi-CPU parallelization
            # Use num_prediction_workers (set at function start) instead of global num_cpu_workers
            
            # Try multi-CPU parallelization if num_prediction_workers > 1 and no GPU
            if num_available_gpus == 0 and num_prediction_workers > 1:
                logger.info(f"Using {num_prediction_workers} CPU cores for parallel prediction")
                
                total_samples = pred_x_tensor.shape[0]
                samples_per_worker = total_samples // num_prediction_workers
                
                # Convert tensor to numpy for pickling
                pred_x_tensor_np = pred_x_tensor.numpy()
                
                # Get model/likelihood state dicts
                model_state_dict = checkpoint['model_state_dict']
                likelihood_state_dict = checkpoint['likelihood_state_dict']
                
                predictions = []
                
                try:
                    logger.info(f"Starting PARALLEL prediction on {num_prediction_workers} CPU cores using ProcessPoolExecutor")
                    
                    with ProcessPoolExecutor(max_workers=num_prediction_workers, mp_context=mp.get_context('spawn')) as executor:
                        futures = []
                        for worker_id in range(num_prediction_workers):
                            start_idx = worker_id * samples_per_worker
                            end_idx = (worker_id + 1) * samples_per_worker if worker_id < num_prediction_workers - 1 else total_samples
                            
                            if start_idx < total_samples:
                                logger.info(f"Submitting CPU worker {worker_id}: samples {start_idx}-{end_idx} ({end_idx-start_idx} samples)")
                                future = executor.submit(_predict_worker, worker_id, start_idx, end_idx, 
                                                       pred_x_tensor_np, batch_size, model_state_dict, 
                                                       likelihood_state_dict, kernel_serialized, is_gpu=False)
                                futures.append((worker_id, future))
                        
                        # Collect results in order
                        for worker_id, future in futures:
                            try:
                                result = future.result()
                                predictions.append(result)
                                logger.info(f"CPU worker {worker_id}: Completed {len(result)} predictions")
                            except Exception as e:
                                logger.error(f"CPU worker {worker_id}: Prediction failed with error: {e}")
                                import traceback
                                logger.error(f"Traceback: {traceback.format_exc()}")
                                raise
                    
                    # Combine predictions
                    y_pred_scaled = np.concatenate(predictions)
                    logger.info(f"PARALLEL multi-CPU prediction completed on {num_prediction_workers} cores")
                except Exception as e:
                    logger.error(f"CPU ProcessPoolExecutor failed: {e}")
                    logger.warning("Falling back to sequential CPU prediction")
                    y_pred_scaled = None  # Mark as failed, will use sequential fallback
            
            # Sequential single GPU/CPU prediction (fallback or when num_prediction_workers=1)
            if y_pred_scaled is None:
                logger.info(f"Entering sequential prediction path (num_prediction_workers={num_prediction_workers}, num_available_gpus={num_available_gpus})")
                
                if num_available_gpus == 1:
                    logger.info("Using single GPU for prediction")
                    device_single = torch.device('cuda:0')
                elif num_available_gpus == 0:
                    logger.info("Using single CPU core for prediction (sequential)")
                    device_single = torch.device('cpu')
                else:
                    logger.info(f"Using CPU for prediction")
                    device_single = torch.device('cpu')
                
                logger.info(f"Reconstructing kernel for device: {device_single}")
                # Reconstruct kernel for single device
                reconstructed_kernel = build_kernel_from_serialized_enhanced(kernel_serialized)
                
                logger.info("Creating dummy data for model initialization")
                dummy_x = torch.randn(10, pred_x_tensor.shape[1]).to(device_single)
                dummy_y = torch.randn(10).to(device_single)
                
                logger.info("Initializing likelihood and model")
                likelihood_single = gpytorch.likelihoods.GaussianLikelihood().to(device_single)
                model_single = ExactGPModel(dummy_x, dummy_y, likelihood_single, 
                                          reconstructed_kernel.to(device_single), set_init=False).to(device_single)
                
                logger.info("Loading model state dict")
                model_single.load_state_dict(checkpoint['model_state_dict'])
                likelihood_single.load_state_dict(checkpoint['likelihood_state_dict'])
                
                model_single.eval()
                likelihood_single.eval()
                
                logger.info(f"Moving prediction data to {device_single}")
                pred_x_tensor = pred_x_tensor.to(device_single)
                
                logger.info(f"Starting sequential prediction of {pred_x_tensor.shape[0]} samples with batch_size={batch_size}")
                predictions = []
                for i in range(0, pred_x_tensor.shape[0], batch_size):
                    batch_end = min(i + batch_size, pred_x_tensor.shape[0])
                    batch = pred_x_tensor[i:batch_end]
                    
                    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                         gpytorch.settings.cholesky_jitter(1e-1):
                        preds = likelihood_single(model_single(batch))
                        y_pred_batch = preds.mean.cpu().numpy()
                        predictions.append(y_pred_batch)
                    
                    if (i // batch_size) % 10 == 0:
                        logger.info(f"Processed {i}/{pred_x_tensor.shape[0]} samples")
                
                logger.info("Concatenating predictions")
                y_pred_scaled = np.concatenate(predictions)
                logger.info(f"Sequential prediction completed: {len(y_pred_scaled)} predictions")

        # Convert back to original scale
        y_pred = y_scaler_enhanced.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Save predictions
        if output_file is None:
            output_file = 'GPR_Enhanced_Predictions_MultiGPU.csv'
        
        pred_df = pd.DataFrame({
            'Prediction': y_pred,
            'Prediction_Scaled': y_pred_scaled
        })
        pred_df.to_csv(output_file, index=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\nPrediction finished --- {elapsed_time} seconds ---\n")
        logger.info(f"Multi-GPU prediction completed in {elapsed_time:.2f} seconds")
        logger.info(f"Predictions saved to {output_file}")
        logger.info(f"Prediction statistics: Mean={np.mean(y_pred):.4f}, Std={np.std(y_pred):.4f}, Min={np.min(y_pred):.4f}, Max={np.max(y_pred):.4f}")
        
        return y_pred, elapsed_time

    except Exception as e:
        logger.error(f"Error in multi-GPU prediction: {e}")
        raise


# Call the enhanced prediction function
# Multiprocessing start method is already set at the top of the file
# This is required for CUDA to work with ProcessPoolExecutor

# After training is complete, run prediction
if __name__ == '__main__':
    try:
        # Pass the same number of workers used in training
        final_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if final_gpu_count == 0:
            # Use the CPU workers count from training
            predictions, pred_time = predict_with_enhanced_model_multi_gpu(num_workers=num_cpu_workers)
        else:
            # Use GPU count
            predictions, pred_time = predict_with_enhanced_model_multi_gpu(num_workers=final_gpu_count)
        logger.info(f"Enhanced multi-GPU prediction completed successfully in {pred_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")