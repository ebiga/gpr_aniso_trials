import silence_tensorflow.auto

import torch
import gpytorch

import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import layers, saving

tf.keras.backend.set_floatx('float64')
torch.set_default_dtype(torch.float64)

tf.random.set_seed(42)
keras.utils.set_random_seed(42)
torch.manual_seed(42)



# GPYTorch loves a class, doesn't it
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def set_hyperparameters(self, lens, vars=None):
        with torch.no_grad():

            # Set the hyperparameters
            self.covar_module.base_kernel.raw_lengthscale.copy_(
                self.covar_module.base_kernel.raw_lengthscale_constraint.inverse_transform(torch.tensor(lens))
                )
            if vars: self.covar_module.raw_outputscale.copy_(
                self.covar_module.raw_outputscale_constraint.inverse_transform(torch.tensor(vars))
                )

        # Force model out of eval mode to recompute internals ...
        self.train()
        self.likelihood.train()

        # And back...
        self.eval()
        self.likelihood.eval()

        return



# Class necessary to expand the input layer shape for the multiheadattention
@saving.register_keras_serializable()
class ExpandALayer(layers.Layer): 
    def call(self, x):
        return tf.expand_dims(x, axis=1)



# Class necessary to squeeze back the output shape of the multiheadattention
@saving.register_keras_serializable()
class SqueezeALayer(layers.Layer):
    def call(self, x):
        return tf.squeeze(x, axis=1)
