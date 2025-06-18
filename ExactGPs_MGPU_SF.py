#!/usr/bin/env python3
#####################################################################
#
#  Copyright 2025
#
#  Zhenqiang Wang
#
#  College of Earth, Ocean, and Atmospheric Sciences
#  Oregon State University
#  Corvallis, OR 97331
#
#  email: zhenqiang.wang@oregonstate.edu
#
# This program is free software; you can redistribute it
# and/or modify it.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#                                                                   
# -*- coding: utf-8 -*-
#                                                                   
#####################################################################

#####################################################################
# Exact GP Regression with Multiple GPUs                     
# https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/Simple_MultiGPU_GP_Regression.html#Normalization-and-train/test-Splits #
#####################################################################

#####################################################################
# Base Python Related Modules                                       #
#####################################################################
import math
import torch
import gpytorch
import sys
from matplotlib import pyplot as plt
sys.path.append('../')
from LBFGS import FullBatchLBFGS

#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

import os
import urllib.request
from scipy.io import loadmat

import numpy as np

import pandas as pd

from math import floor
from sklearn.preprocessing import StandardScaler

#####################################################################
# Import Dataset                                                    #
#####################################################################

# a small number of data points for training and testing
train_x_df = pd.read_csv('Forcing_750sim_SF.csv')
# Convert DataFrame to a NumPy array
numpy_array_x = train_x_df.values
# Convert NumPy array to a PyTorch tensor
X_ft = torch.from_numpy(numpy_array_x)
# Convert to double
X = X_ft.to(torch.float32)

train_y_df = pd.read_csv('TWL_750sim_SF.csv')
# Convert DataFrame to a NumPy array
numpy_array_y = train_y_df.values
# Convert NumPy array to a PyTorch tensor
y_ft = torch.from_numpy(numpy_array_y)
y_ft.resize_(y_ft.size()[0])
# Convert to double
y = y_ft.to(torch.float32)

# a large number of data points for prediction
predict_x_df = pd.read_csv('Inputs_GPR_predict_10yr_sim_1.csv')
# Convert DataFrame to a NumPy array
numpy_array_predict_x = predict_x_df.values
# Convert NumPy array to a PyTorch tensor
predict_X_ft = torch.from_numpy(numpy_array_predict_x)
# Convert to double
Xp = predict_X_ft.to(torch.float32)

#####################################################################
# Normalization and train/test Split                                #
#####################################################################
#Normalize
X_scaler = StandardScaler()
y_scaler = StandardScaler()
Xp_scaler = StandardScaler()

X_norm = X_scaler.fit_transform(X)
y_norm = y_scaler.fit_transform(y.reshape(-1,1)).ravel()
Xp_norm = Xp_scaler.fit_transform(Xp)

# convert to tensor
X_t = torch.tensor(X_norm,dtype=torch.float32)
y_t = torch.tensor(y_norm,dtype=torch.float32)
Xp_t = torch.tensor(Xp_norm,dtype=torch.float32)

train_n = int(floor(0.8 * len(X)))
train_x = X_t[:train_n, :]
train_y = y_t[:train_n]
predict_x = Xp_t

test_x = X_t[train_n:, :]
test_y = y_t[train_n:]

# normalize features
mean = train_x.mean(dim=-2, keepdim=True)
std = train_x.std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

mean = predict_x.mean(dim=-2, keepdim=True)
std = predict_x.std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
predict_x = (predict_x - mean) / std

# normalize labels
mean, std = train_y.mean(),train_y.std()
train_y = (train_y - mean) / std
test_y = (test_y - mean) / std

# make continguous
train_x, train_y = train_x.contiguous(), train_y.contiguous()
test_x, test_y = test_x.contiguous(), test_y.contiguous()
predict_x = predict_x.contiguous()

output_device = torch.device('cuda:0')

train_x, train_y = train_x.to(output_device), train_y.to(output_device)
test_x, test_y = test_x.to(output_device), test_y.to(output_device)
predict_x = predict_x.to(output_device)

#####################################################################
# How many GPUs do you want to use?                                 #
#####################################################################

n_devices = torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))

#####################################################################
# Exact GP Regression with Multiple GPUs                            #
#####################################################################

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_devices):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train(train_x,
          train_y,
          n_devices,
          output_device,
          preconditioner_size,
          n_training_iter,
):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
    model = ExactGPModel(train_x, train_y, likelihood, n_devices).to(output_device)
    model.train()
    likelihood.train()

    optimizer = FullBatchLBFGS(model.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


    with gpytorch.settings.max_preconditioner_size(preconditioner_size):

        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)            
            return loss

        loss = closure()
        loss.backward()

        for i in range(n_training_iter):
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, _, _, _, _, _, fail = optimizer.step(options)

            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, n_training_iter, loss.item(),
                model.covar_module.module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))

            if fail:
                print('Convergence reached!')
                break

    print(f"Finished training on {train_x.size(0)} data points using {n_devices} GPUs.")
    return model, likelihood


#####################################################################
# Training                                                          #
#####################################################################

model, likelihood = train(train_x, train_y,
                          n_devices=n_devices, output_device=output_device,
                          preconditioner_size=100,
                          n_training_iter=20)


#####################################################################
# Computing test time caches                                        #
#####################################################################

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Make predictions on a small number of test points to get the test time caches computed
    latent_pred = model(test_x[:2, :])
    del latent_pred  # We don't care about these predictions, we really just want the caches.


#####################################################################
# Testing: Computing predictions using a small number of data points#
#####################################################################

with torch.no_grad(), gpytorch.settings.fast_pred_var():
     latent_pred = model(test_x)

test_rmse = torch.sqrt(torch.mean(torch.pow(latent_pred.mean - test_y, 2)))
print(f"Test RMSE: {test_rmse.item()}")

#####################################################################
# Prediction: Computing predictions using a large number of data points#
#####################################################################
start_time = time.time()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
     latent_pred = model(predict_x)

end_time = time.time()
prediction_time = end_time - start_time
print(f"Prediction time: {prediction_time:.4f} seconds")

