**Multi-GPU / Multi-CPU Exact Gaussian Process Regression (GPR)

with Parallel Cross-Validation**

This repository provides an accelerated implementation of Exact Gaussian Process Regression (GPR) using GPyTorch and PyTorch, designed for parallel training and prediction on multiple GPUs or multiple CPU cores.
The code is built to support large scientific datasets—such as long-duration flood driver time series—and enables fast, scalable surrogate modeling for environmental and climate applications.

The implementation serves as a high-performance alternative to MATLAB-based CPU GPR workflows and has been tested on real-world 500-year compound coastal flooding datasets.

Key Features

Multi-GPU parallelism for both training and inference

Multi-CPU support for environments without GPUs

Full cross-validation in parallel (across GPUs or CPU workers)

Automatic hardware detection (all GPUs, single GPU, selected GPU IDs, or CPU)

Supports very large datasets using GPyTorch's lazy tensors and memory-efficient kernels

Automatic batch size tuning and memory cleanup

Robust fallback behavior when GPUs are not available

Compatible with Linux, Windows, and HPC clusters

File Description
ExactGPs_GPU.py

The main script implementing:

Exact GPR training using GPyTorch

Multi-GPU and multi-CPU execution

Parallel k-fold cross-validation

Enhanced prediction using a trained multi-GPU model

Logging, error handling, and automatic resource assignment

Installation
1. Clone the repository
git clone https://github.com/<your-repo>/ExactGPs_GPU.git
cd ExactGPs_GPU

2. Install dependencies
pip install torch gpytorch pandas numpy scikit-learn


(For GPU users, install the CUDA-enabled PyTorch version appropriate for your system.)

Usage
Interactive run
python ExactGPs_GPU.py


When prompted:

Hardware selection options

Use all GPUs

Use a single GPU

Specify selected GPU IDs

Use CPU only

CPU mode prompt

Enter number of CPU cores to use (default = all available)

Hardware Modes
Multi-GPU

Automatically detects GPU count

Trains each CV fold on a separate GPU

Performs parallel prediction

Single-GPU

Runs full training sequentially on the specified GPU

Multi-CPU

Parallel CV and prediction via multiprocessing

Single-CPU

Fully sequential, slowest mode

Provided for compatibility

Input / Output
Inputs

CSV files for:

Features (e.g., flood drivers)

Targets (e.g., total water levels)

Outputs

Trained GPR model

Prediction arrays

Log files including:

Hardware selection

Training duration

Cross-validation metrics

Prediction timing

Modeling Details

Exact GPR using RBF kernels, Matérn kernels, or custom kernels

Leverages GPyTorch lazy tensors (no DataParallel)

Includes memory-optimized data loaders

Supports large-scale hydrodynamic surrogate modeling workflows

Example Application

The script has been used for:

High-resolution compound flood prediction

Multi-fidelity surrogate modeling

Long-duration environmental time series emulation

Calibration and skill assessment of diffusion-based flood surrogates

References

This work builds on the GPU-accelerated GPR concepts developed in earlier MATLAB implementations:

Wang et al., Hybrid Statistical–Dynamical Framework for Compound Coastal Flooding

Wang et al., Surrogate Modeling for Probabilistic Flood Hazard Assessment

Contact

Author: Zhenqiang Wang
Affiliation: Oregon State University
Email: zhenqiang.wang@oregonstate.edu
