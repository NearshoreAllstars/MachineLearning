# -*- coding: utf-8 -*-
"""
Created on 9/16, 2025

@author: Zhenqiang Wang

# Here’s a GPU-accelerated Python code using GPyTorch and PyTorch for k-fold cross-validation
# of an Exact Gaussian Process Regression (GPR) model. It allows a full grid search over all 
# kernel, subkernel, optimizer, and training hyperparameters—including for composite kernels (sums/products),
# evaluates RMSE and R², and uses the best model for prediction.

# GitHub Copilot

# You'll need:
# gpytorch, torch, scikit-learn, and a CUDA-capable GPU.

# Notes:
# Replace the example data with your own.
# You can add more kernels or optimizers as needed.
# For large datasets, consider using approximate GPs (e.g., gpytorch.models.ApproximateGP).

# Runtime: ~xx mins on CPU, ~yy mins on GPU if comparing all kernels and optimizers (depends on hardware and dataset size)
"""
import torch
import gpytorch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import os
import matplotlib.pyplot as plt
from itertools import product
import joblib

# give a fixed seed for full reproducibility
seed_id = 42
torch.manual_seed(seed_id)
np.random.seed(seed_id)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

################################################################################################################
# ExactGPModel class
# This class defines the Exact GP model with a constant mean and a specified kernel.
################################################################################################################
# Add/Strengthen Constraints and Initializations
# This function recursively sets constraints and initial values for all sub-kernels in the kernel tree, especially
# for the optimizer (especially LBFGS), which may push parameters into unstable regions.

# If Cholesky decomposition fails, try to Constrain and Initialize Kernel Hyperparameters 
# by trying different initial values, especially a much wider interval (e.g., Interval(0.1, 20.0)) 
# on kernel hyperparametersto allow more flexibility/more accuracy
def _set_constraints_and_initial_values(kernel):
    if hasattr(kernel, "base_kernel"):
        _set_constraints_and_initial_values(kernel.base_kernel)
    if isinstance(kernel, (gpytorch.kernels.AdditiveKernel, gpytorch.kernels.ProductKernel)):
        for k in kernel.kernels:
            _set_constraints_and_initial_values(k)
    if hasattr(kernel, "raw_lengthscale"):
        kernel.lengthscale = 1.0
        kernel.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(0.1, 20.0))
    if hasattr(kernel, "raw_outputscale"):
        kernel.outputscale = 1.0
        kernel.register_constraint("raw_outputscale", gpytorch.constraints.Interval(0.01, 10.0))
        
# When training, use set_init=True (default).
# When loading for prediction, use set_init=False so do NOT reset the kernel parameters.
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, set_init=True):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        if set_init:
            _set_constraints_and_initial_values(self.covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
################################################################################################################
# loading data
# Make sure there are no NaNs or Infs in your input or output data.
# Check for duplicate rows or columns with zero variance. These cases may cause issues in training (e.g., NaNs).
################################################################################################################
#change the working directory to the script’s folder before any file operation
os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_x_df = pd.read_csv('Forcing_750sim_SF.csv', header=None)
# Convert DataFrame to a NumPy array
numpy_array_x = train_x_df.values
# Convert NumPy array to a PyTorch tensor
X_ft = torch.from_numpy(numpy_array_x)
# Convert to double
X_ftf = X_ft.to(torch.float32)

train_y_df = pd.read_csv('TWL_750sim_SF.csv', header=None)
# Convert DataFrame to a NumPy array
numpy_array_y = train_y_df.values
# Convert NumPy array to a PyTorch tensor
y_ft = torch.from_numpy(numpy_array_y)
y_ftf = y_ft.reshape(-1,1).to(torch.float32)

# convert degree to radians before scaling, this may be important for angular features
degree_columns = [i for i in range(10)]
X_ftf_np = X_ftf.numpy() if isinstance(X_ftf, torch.Tensor) else X_ftf
X_ftf_np[:, degree_columns] = np.deg2rad(X_ftf_np[:, degree_columns])
X_ftf = torch.from_numpy(X_ftf_np)

#Normalize
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X_ftf)
y = y_scaler.fit_transform(y_ftf.reshape(-1,1)).ravel()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Check Data for NaNs/Infs and Duplicates. Before training, ensure your data has no NaNs, Infs, or duplicate rows:
assert not np.isnan(X).any() and not np.isnan(y).any(), "NaNs in data!"
assert not np.isinf(X).any() and not np.isinf(y).any(), "Infs in data!"
assert np.unique(X, axis=0).shape[0] == X.shape[0], "Duplicate rows in X!"

################################################################################################################
# Define training hyperparameters to train/optimize the model
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

def train_and_eval(train_x, train_y, test_x, test_y, kernel, optimizer_name, lr, max_epochs):
    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(train_x, train_y, likelihood, kernel).to(device)

    model.train()
    likelihood.train()

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    elif optimizer_name == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    losses = []
    best_loss = float('inf')
    patience_counter = 0

    for i in range(max_epochs):
        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            return loss

        if optimizer_name == 'LBFGS':
            loss = optimizer.step(closure)
            loss = loss.item() if hasattr(loss, 'item') else float(loss)
        else:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            loss = loss.item()
        losses.append(loss)

        # Early stopping check
        if loss < best_loss - early_stopping_delta:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {i+1} (loss stabilized)")
            break
    
    # # Optionally plot the loss curve for the last fold
    # if max_epochs == epoch_options[-1] and lr == learning_rate_options[-1]:
    #     plt.plot(losses)
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Negative Marginal Log Likelihood')
    #     plt.title('Loss Curve')
    #     plt.show(block=False)
    #     pytime.sleep(5)
    #     plt.close()

    model.eval()
    likelihood.eval()

    # Adds jitter for Cholesky stability. May need to increase jitter if the model is unstable.
    # with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-2):
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-1):
        preds = likelihood(model(test_x))
        y_pred = preds.mean
        
        # Convert Both y_true and y_pred back to meters using the GLOBAL scaler!
        y_true_z = test_y.cpu().numpy().reshape(-1,1)
        y_pred_z = y_pred.cpu().numpy().reshape(-1,1)
        
        y_true_m = y_scaler.inverse_transform(y_true_z).ravel()
        y_pred_m = y_scaler.inverse_transform(y_pred_z).ravel()
            
        rmse = np.sqrt(mean_squared_error(y_true_m, y_pred_m))
        r2 = r2_score(y_true_m, y_pred_m)
    return rmse, r2, model, likelihood, losses


################################################################################################################
# Define kernels and kernel parameters to train/optimize the model
################################################################################################################
# Define a broad set of commonly-used kernels, RBF, Matern, Polynomial, Linear, etc.
# Composite kernels can be unstable if the data is not suitable for the kernel, so may just use a single kernel.
# ensure the kernel structure is identical during training and prediction
# sums the kernels, then applies a single outputscale rather than sums two independently scaled kernels
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


# Define kernel hyperparameter grids to search over. Try a wider range (and possibly finer steps)
# note: These hyperparameters (e.g., lengthscale) are used as the starting point for all lengthscales, 
# but they are not fixed—they will be optimized (e.g., via ARD) during training.
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

# Recursively extract all subkernels and their types
def extract_kernel_types(kernel):
    """
    Recursively extract all base kernel types from a (possibly composite) kernel.
    Returns a list of (path, type) where path is a tuple of indices to reach the subkernel.
    """
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
    
    
# Recursively build a kernel from a structure and a param dict
# All supported kernels will use ARD (ARD allows each input dimension to have its own lengthscale, 
# which is crucial for high-dimensional data and when input features have different units/scales).This will improve model accuracy.
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
# Define oprtimizers and optimizer parameters to train/optimize the model
################################################################################################################

# Define a broad set of commonly-used optimizers, LBFGS (LBFGS can be unstable for GPs!), Adam, SGD
# You can use PyTorch's LBFGS (unconstrained), but not L-BFGS-B (with bounds) in this code.
# For true L-BFGS-B with bounds, you would need to use SciPy's optimizer and manually handle
# parameter updates, which is not straightforward with GPyTorch models.

optimizer_options = [
    'Adam',
    # 'SGD',
    # 'RMSprop',
    # 'Adagrad',
    # 'AdamW',
    # 'Adadelta',
    # 'LBFGS'
]
# Define optimizer hyperparameter grids
optimizer_hyperparams = {
    'Adam': {'lr': learning_rate_options, 'weight_decay': [0, 1e-4]},
    'SGD': {'lr': learning_rate_options, 'momentum': [0.0, 0.9]},
    'RMSprop': {'lr': learning_rate_options},
    'Adagrad': {'lr': learning_rate_options},
    'AdamW': {'lr': learning_rate_options, 'weight_decay': [0, 1e-4]},
    'Adadelta': {'lr': learning_rate_options},
    'LBFGS': {'lr': learning_rate_options}
}

# build_kernel_from_option is used during training and grid search to 
# construct kernels from your kernel_options and hyperparameter grids.
def build_kernel_from_option(kernel_option, params):
    # For sum/product kernels, just use as is (no hyperparam grid search)
    if isinstance(kernel_option, (gpytorch.kernels.AdditiveKernel, gpytorch.kernels.ProductKernel)):
        return kernel_option
    if isinstance(kernel_option, gpytorch.kernels.ScaleKernel):
        base = kernel_option.base_kernel
        base_type = type(base).__name__
        if base_type == 'RBFKernel':
            k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale=params.get('lengthscale', 1.0)))
        elif base_type == 'MaternKernel':
            k = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=params.get('nu', 2.5), lengthscale=params.get('lengthscale', 1.0))
            )
        elif base_type == 'PolynomialKernel':
            k = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PolynomialKernel(power=params.get('power', 2), offset=params.get('offset', 0.0))
            )
        elif base_type == 'LinearKernel':
            k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        elif base_type == 'PeriodicKernel':
            k = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PeriodicKernel(period_length=params.get('period_length', 1.0))
            )
        elif base_type == 'SpectralMixtureKernel':
            k = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.SpectralMixtureKernel(
                    num_mixtures=params.get('num_mixtures', 2),
                    ard_num_dims=params.get('ard_num_dims', X.shape[1])
                )
            )
        else:
            # If the base kernel is itself an AdditiveKernel or ProductKernel, return as is
            if isinstance(base, (gpytorch.kernels.AdditiveKernel, gpytorch.kernels.ProductKernel)):
                return kernel_option
            raise ValueError(f"Unknown kernel type: {base_type}")
    else:
        # For other kernels, just use as is
        k = kernel_option
    return k
    
def build_optimizer(model, opt_name, opt_params):
    if opt_name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=opt_params['lr'], weight_decay=opt_params.get('weight_decay', 0))
    elif opt_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=opt_params['lr'], momentum=opt_params.get('momentum', 0))
    elif opt_name == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=opt_params['lr'])
    elif opt_name == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=opt_params['lr'])
    elif opt_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=opt_params['lr'], weight_decay=opt_params.get('weight_decay', 0))
    elif opt_name == 'Adadelta':
        return torch.optim.Adadelta(model.parameters(), lr=opt_params['lr'])
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

################################################################################################################
# Training and testing on GPU, looping over sample sizes, kernels, optimizers, and hyperparameters
# calculate RMSE and R² for each size based on the best model found via grid search, plot convergence, 
# save only the best model and scalers for prediction
# K-Fold Cross Validation with grid search for learning rate and epochs
################################################################################################################
# used only during model loading/prediction to reconstruct the kernel 
# from the serialized structure you saved
# Serialize kernels:
def serialize_kernel(kernel):
    # Handle ScaleKernel wrapping
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
            'params': {'lengthscale': kernel.lengthscale.detach().cpu().numpy().tolist()}
        }
    elif isinstance(kernel, gpytorch.kernels.MaternKernel):
        return {
            'kernel_type': 'MaternKernel',
            'params': {
                'nu': float(kernel.nu) if hasattr(kernel, 'nu') else 2.5,
                'lengthscale': kernel.lengthscale.detach().cpu().numpy().tolist()
            }
        }
    elif isinstance(kernel, gpytorch.kernels.PolynomialKernel):
        return {
            'kernel_type': 'PolynomialKernel',
            'params': {
                'power': int(kernel.power) if hasattr(kernel, 'power') else 2,
                'offset': float(kernel.offset) if hasattr(kernel, 'offset') else 0.0,
                'lengthscale': kernel.lengthscale.detach().cpu().numpy().tolist() if hasattr(kernel, 'lengthscale') else None
            }
        }
    elif isinstance(kernel, gpytorch.kernels.LinearKernel):
        return {
            'kernel_type': 'LinearKernel',
            'params': {
                'variance': kernel.variance.detach().cpu().numpy().tolist() if hasattr(kernel, 'variance') else None
            }
        }
    elif isinstance(kernel, gpytorch.kernels.PeriodicKernel):
        return {
            'kernel_type': 'PeriodicKernel',
            'params': {
                'period_length': kernel.period_length.detach().cpu().numpy().tolist() if hasattr(kernel, 'period_length') else 1.0,
                'lengthscale': kernel.lengthscale.detach().cpu().numpy().tolist() if hasattr(kernel, 'lengthscale') else None
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
        raise ValueError(f"Unknown kernel type: {type(kernel)}")
    
sample_sizes = [50, 150, 250, 350, 450, 550, 650, 750]
# sample_sizes = [50, 150]
# sample_sizes = [750]
# --- Collect all fold results for boxplot ---
rmse_folds_by_size = []
r2_folds_by_size = []

for size_idx, size in enumerate(sample_sizes):
    print(f"\n--- Cross-validation and grid search for Sample Size: {size} ---")
    X_subset = X[:size]
    y_subset = y[:size]
    # K-Fold Cross Validation with grid search for learning rate and epochs
    kf = KFold(n_splits=5, shuffle=True, random_state=seed_id)
    results = []
    start_time = time.time()

    for kernel_option in kernel_options:
        # 1. Extract all subkernel types and their paths
        subkernel_types = extract_kernel_types(kernel_option)
        # 2. Build a grid for each subkernel
        subkernel_param_grids = []
        for path, ktype in subkernel_types:
            grid = kernel_hyperparams.get(ktype, {})
            if grid:
                keys, values = zip(*grid.items())
                combos = [dict(zip(keys, v)) for v in product(*values)]
            else:
                combos = [{}]
            subkernel_param_grids.append(combos)
        # 3. Build all combinations of subkernel params (as a dict mapping path -> params)
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
                    # will try all the values of lr in opt_params
                    for lr in [opt_params.get('lr', 0.01)]:
                        for epochs in epoch_options:
                            rmses, r2s = [], []
                            for fold, (train_idx, test_idx) in enumerate(kf.split(X_subset), 1):
                                rmse, r2, _, _, _ = train_and_eval(
                                    X_subset[train_idx], y_subset[train_idx], X_subset[test_idx], y_subset[test_idx],
                                    kernel, opt, lr, epochs
                                )
                                rmses.append(rmse)
                                r2s.append(r2)
                                print(f"Fold {fold} RMSE: {rmse:.4f} m, R²: {r2:.4f}")
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
                                'learning_rate': lr,
                                'epochs': epochs
                            })

    # Find best model (lowest RMSE) for this sample size
    best = min(results, key=lambda x: x['rmse'])
    best['rmse_folds'] = rmses  # Add this line
    best['r2_folds'] = r2s      # Add this line
    print(f"Best combination for sample size {size}:", best)
    rmse_folds_by_size.append(best.get('rmse_folds', []))
    r2_folds_by_size.append(best.get('r2_folds', []))

    # Only save the model for the largest sample size
    if size_idx == len(sample_sizes) - 1:
        # Retrain on full data with best kernel/optimizer/hyperparameters
        best_kernel = build_kernel_with_params(best['kernel_option'], best['kernel_param_dict'])
        best_lr = best['learning_rate']
        best_epochs = best['epochs']
        best_opt = best['optimizer']
        final_rmse, final_r2, final_model, final_likelihood, final_losses = train_and_eval(
            X_subset, y_subset, X_subset, y_subset, best_kernel, best_opt, best_lr, best_epochs
        )
        # Save the serialized kernel structure when saving model
        kernel_serialized = serialize_kernel(best_kernel)
        model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_gpytorch_model.pth")
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'likelihood_state_dict': final_likelihood.state_dict(),
            'kernel_serialized': kernel_serialized,
            'kernel_param_dict': best['kernel_param_dict'],
            'optimizer': best_opt,
            'optimizer_params': best['optimizer_params'],
            'kernel_option_str': str(best['kernel_option']),
        }, model_save_path)
        # Save scalers with joblib
        joblib.dump(X_scaler, os.path.join(os.path.dirname(os.path.abspath(__file__)), "X_scaler.save"))
        joblib.dump(y_scaler, os.path.join(os.path.dirname(os.path.abspath(__file__)), "y_scaler.save"))
        print(f"Best model for largest sample size saved to {model_save_path}")

    end_time = time.time()
    print(f"Sample size {size} finished --- {end_time - start_time:.2f} seconds ---")

# Save RMSE and R² results to CSV
pd.DataFrame({'SampleSize': sample_sizes, 'RMSE': rmse_folds_by_size, 'R2': r2_folds_by_size}).to_csv("GPR_RMSE_R2_by_SampleSize.csv", index=False)
print("✅ RMSE and R² results saved to CSV file.")

# --- Plotting boxplots ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.boxplot(rmse_folds_by_size, labels=sample_sizes)
plt.xlabel("Sample Size", fontsize=12)
plt.ylabel("RMSE (m)", fontsize=12)
plt.title("RMSE across Sample Sizes", fontsize=14)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.boxplot(r2_folds_by_size, labels=sample_sizes)
plt.xlabel("Sample Size", fontsize=12)
plt.ylabel("R²", fontsize=12)
plt.title("R² across Sample Sizes", fontsize=14)
plt.grid(True)

plt.suptitle("GPR Model Performance Metrics", fontsize=16)
plt.tight_layout()

fig_dir = r"C:\Users\wangzh3\OneDrive - Oregon State University\Cascadia CoPes Hub\AI_GPU\NVIDIA\Run GPR on GPU\Fig"
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "GPR_RMSE_R2_by_SampleSize.png")
plt.savefig(fig_path)
plt.show()
print(f"✅ Box plots saved to {fig_path}")

################################################################################################################
# Predict with the best model
################################################################################################################
# Load the best model and use it to predict
start_time = time.time()

checkpoint = torch.load(model_save_path, map_location=device)
print("Kernel option (string):", checkpoint.get('kernel_option_str', 'Not saved'))

# Load scalers
X_scaler = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "X_scaler.save"))
y_scaler = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "y_scaler.save"))

# Rebuild the kernel. All supported kernels will use ARD (ARD allows each input dimension to have its own lengthscale, 
# which is crucial for high-dimensional data and when input features have different units/scales).This will improve model accuracy.
def build_kernel_from_serialized(kernel_dict):
    kernel_type = kernel_dict['kernel_type']
    if kernel_type == 'ScaleKernel':
        return gpytorch.kernels.ScaleKernel(build_kernel_from_serialized(kernel_dict['base_kernel']))
    elif kernel_type == 'AdditiveKernel':
        subkernels = [build_kernel_from_serialized(sub) for sub in kernel_dict['subkernels']]
        return gpytorch.kernels.AdditiveKernel(*subkernels)
    elif kernel_type == 'ProductKernel':
        subkernels = [build_kernel_from_serialized(sub) for sub in kernel_dict['subkernels']]
        return gpytorch.kernels.ProductKernel(*subkernels)
    elif kernel_type == 'RBFKernel':
        return gpytorch.kernels.RBFKernel(lengthscale=kernel_dict['params'].get('lengthscale', 1.0), ard_num_dims=X.shape[1])
    elif kernel_type == 'MaternKernel':
        return gpytorch.kernels.MaternKernel(
            nu=kernel_dict['params'].get('nu', 2.5),
            lengthscale=kernel_dict['params'].get('lengthscale', 1.0),
            ard_num_dims=X.shape[1]
        )
    elif kernel_type == 'PolynomialKernel':
        return gpytorch.kernels.PolynomialKernel(
            power=kernel_dict['params'].get('power', 2),
            offset=kernel_dict['params'].get('offset', 0.0),
            ard_num_dims=X.shape[1]
        )
    elif kernel_type == 'SpectralMixtureKernel':
        return gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=kernel_dict['params'].get('num_mixtures', 2),
            ard_num_dims=kernel_dict['params'].get('ard_num_dims', X.shape[1])
        )
    elif kernel_type == 'LinearKernel':
        return gpytorch.kernels.LinearKernel(ard_num_dims=X.shape[1])
    elif kernel_type == 'PeriodicKernel':
        return gpytorch.kernels.PeriodicKernel(
            period_length=kernel_dict['params'].get('period_length', 1.0),
            ard_num_dims=X.shape[1]
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

# Prepare kernel and likelihood
kernel_serialized = checkpoint['kernel_serialized']
kernel = build_kernel_from_serialized(kernel_serialized)
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

# Prepare model
# When loading the model for prediction, must pass the original training data to the model constructor:
final_model = ExactGPModel(
    torch.tensor(X, dtype=torch.float32).to(device),
    torch.tensor(y, dtype=torch.float32).to(device),
    likelihood, kernel, set_init=False  # <--- IMPORTANT
).to(device)
final_model.load_state_dict(checkpoint['model_state_dict'])
likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

final_model.eval()
likelihood.eval()

# Load and normalize new data
test_xx_df = pd.read_csv('Inputs_GPR_predict_1yr_sim_1.csv', header=None)
numpy_array_test_xx = test_xx_df.values
test_XX = torch.from_numpy(numpy_array_test_xx).to(torch.float32)

# convert degree to radians for tidal constituents before scaling, need to do the same transformation as training data
degree_columns = [i for i in range(10)]
test_XX_np = test_XX.numpy() if isinstance(test_XX, torch.Tensor) else test_XX
test_XX_np[:, degree_columns] = np.deg2rad(test_XX_np[:, degree_columns])
test_XX = torch.from_numpy(test_XX_np)

test_XX_norm = X_scaler.transform(test_XX)
X_new_torch = torch.tensor(test_XX_norm, dtype=torch.float32).to(device)

# Predict, adds jitter for Cholesky stability.
# with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-2):
with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-1):
    preds = likelihood(final_model(X_new_torch))
    pred_mean = preds.mean.cpu().numpy()
    # Inverse transform to original scale
    pred_mean_original = y_scaler.inverse_transform(pred_mean.reshape(-1, 1)).ravel()
    print("Predictions (original scale):", pred_mean_original)

# # The commented out section below can be used to check how well the model fits the training data 
# # (whether the prediction is correct or underfits or overfits).
# #  Try Predicting on Training Data, just to see how well it fits the training data-debugging
# with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-1):
#     preds_train = likelihood(final_model(torch.tensor(X, dtype=torch.float32).to(device)))
#     pred_mean_train = preds_train.mean.cpu().numpy()
#     pred_mean_train_original = y_scaler.inverse_transform(pred_mean_train.reshape(-1, 1)).ravel()
#     print("First 5 predictions on training data:", pred_mean_train_original[:5])
    
# # Inverse-transform to original scale
# y_true_original = y_scaler.inverse_transform(y.reshape(-1, 1)).ravel()

# print("First 5 true y values (original scale):", y_true_original[:5])
# print("First 5 predicted y values (original scale):", pred_mean_train_original[:5])

# rmse_train = np.sqrt(mean_squared_error(y_true_original, pred_mean_train_original))
# r2_train = r2_score(y_true_original, pred_mean_train_original)
# print(f"Training RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")

end_time = time.time()
execution_time = end_time - start_time
print("Prediction finished --- %s seconds ---" % (execution_time))