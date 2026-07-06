import os
import hjson
import numpy as np
import pandas as pd
import silence_tensorflow.auto
import gpflow
import torch

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.utils import get_custom_objects

# set floats and randoms
gpflow.config.set_default_float('float64')
tf.keras.backend.set_floatx('float64')
torch.set_default_dtype(torch.float64)

tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
torch.manual_seed(42)




## Function to write out kernel params
def generate_kernel_info(model):
    module = type(model).__module__
    
    if "gpflow" in module:
        def kernel_to_code(k):
            # Handle kernel combinations (Sum, Product)
            if isinstance(k, gpflow.kernels.Sum):
                return " + ".join(kernel_to_code(sub_k) for sub_k in k.kernels)
            elif isinstance(k, gpflow.kernels.Product):
                return " * ".join(kernel_to_code(sub_k) for sub_k in k.kernels)
            # Handle individual kernels (e.g., RBF, Constant, etc.)
            params = []
            # Extract parameter name and transformed value
            #_ variance
            param_nam = "var"
            param_val = k.variance.numpy().round(decimals=3)
            params.append(f"{param_nam}={param_val}")
            #_ lengthscales
            param_nam = "len"
            param_val = k.lengthscales.numpy().round(decimals=3)
            params.append(f"{param_nam}={param_val}")
            #_ alpha if so
            if isinstance(k, gpflow.kernels.RationalQuadratic):
                param_nam = "alpha"
                param_val = k.alpha.numpy().round(decimals=3)
                params.append(f"{param_nam}={param_val}")
            # Construct kernel initialization code
            kernel_name = type(k).__name__
            return f"{kernel_name}({', '.join(params)})"
        # Start recursive kernel generation
        return str(kernel_to_code(model.kernel))




## Write predicts out
def write_predicts_file(location, params_in, func_in, func_pred):
    predfile = pd.DataFrame(params_in)
    predfile['f'] = func_in
    predfile['f_pred'] = func_pred
    predfile.to_csv(os.path.join(location, 'test_data_out.csv'), index=False)




## Compute the rms of the mean
def check_mean(atest, mean, refd):
    delta = refd - mean

    rms_check = np.sqrt( np.mean( delta**2. ) )
    mae_check = np.mean( np.abs(delta) )
    max_check = np.max( np.abs(delta) )

    msg = atest + " errors: rms, mean, max: " + f"\t{rms_check:.3e};\t {mae_check:.3e};\t {max_check:.3e}\n"
    print(msg)
    return msg




## My wrapper of predict functions
def my_predicts(model, X):
    module = type(model).__module__
    
    if "gpflow" in module:
        return model.predict_f(X)[0].numpy().reshape(-1)
    
    elif "tensorflow" in module or "keras" in module:
        return model.predict(X).reshape(-1)

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")




## Will define what to do with the kernel variance
def kernel_variance_whatabouts(jsonfile):

    # Get either a false or a value
    kernel_variance = jsonfile['GPR_setup']['kernel_variance']

    if kernel_variance:
        # Variance is fixed
        vars = kernel_variance**2.
        if_train_variance = False

    else:
        # Variance will be optimised, we chose to start from 1
        vars = 1.
        if_train_variance = True

    return vars, if_train_variance




## Will define what to do with the kernel lengthscale
def kernel_lengthscale_whatabouts(jsonfile, Ndimensions):

    # Get either a false or a value
    lens = float(jsonfile['GPR_setup']['kernel_lengthscale'])
    if_ARD = jsonfile['GPR_setup']['if_kernel_lengthscale_ARD']

    Ndim = Ndimensions

    if if_ARD: lens = np.full(Ndim, lens)

    return lens




## Update kernel parameters
def update_kernel_params(model, new_lengthscale, new_variance=None):
    module = type(model).__module__

    #_ GPflow
    if "gpflow" in module:
        model.kernel.lengthscales.assign(new_lengthscale)
        if new_variance: model.kernel.variance.assign(new_variance)
        return




## NN trunk constructor
def build_nn_trunk(input_tensor, nn_layers):
    x = input_tensor
    for nn in nn_layers:
        x = layers.Dense(nn, activation=LeakyELU(beta=0.4), kernel_initializer=keras.initializers.GlorotUniform(seed=None))(x)
    return layers.Dense(1)(x)




# Custom activation function, this is like a leaky ELU
class LeakyELU(tf.keras.layers.Layer):
    def __init__(self, beta=0.4, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        pos = tf.where(inputs > 0, inputs, tf.zeros_like(inputs))
        neg = tf.where(
            inputs <= 0,
            tf.exp((1.0 - self.beta) * inputs) - 1.0 + self.beta * inputs,
            tf.zeros_like(inputs)
        )
        return pos + neg

    def get_config(self):
        config = super().get_config()
        config.update({'beta': self.beta})
        return config

# Register so it can be used in model configs
get_custom_objects().update({'LeakyELU': LeakyELU})
