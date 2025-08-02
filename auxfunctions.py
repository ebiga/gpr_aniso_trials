import os
import hjson
import numpy as np
import pandas as pd
import sklearn
import silence_tensorflow.auto
import gpflow
import torch
import gpytorch

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.utils import get_custom_objects
from keras_tuner import HyperParameters

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
    
    if "sklearn" in module:
        return str(model.kernel_)
    
    elif "gpflow" in module:
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
    
    elif isinstance(model, gpytorch.models.GP):
        msg = "Lengthscale: " + str(model.covar_module.base_kernel.lengthscale.squeeze().tolist()) + "\n" \
            + "Variance: " + str(model.covar_module.outputscale.item()) + "\n"
        return msg




## Reshape due to csv XY and my lovely IJ orders
def reshape_flatarray_like_reference_meshgrid(offending_array, dashape, select_dimension):
    # the csv comes in the reversed order of the IJ mesh grid
    # the flattened array is reshaped into its mesh shape than tranposed to the IJ shape
    ashape = dashape[::-1]
    if select_dimension == '3D':
        return offending_array.reshape(ashape).transpose(2, 1, 0)
    elif select_dimension == '2D':
        return offending_array.reshape(ashape).transpose()

def tf_reshape_flatarray_like_reference_meshgrid(array, dashape, select_dimension):
    ashape = dashape[::-1]
    if select_dimension == '3D':
        reshaped = tf.reshape(array, ashape)
        return tf.transpose(reshaped, perm=[2, 1, 0])
    elif select_dimension == '2D':
        reshaped = tf.reshape(array, ashape)
        return tf.transpose(reshaped)




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
    
    if "sklearn" in module:
        return model.predict(X, return_cov=False)
    
    elif "gpflow" in module:
        return model.predict_f(X)[0].numpy().reshape(-1)
    
    elif "tensorflow" in module or "keras" in module:
        return model.predict(X).reshape(-1)
    
    elif isinstance(model, gpytorch.models.GP):
        model.eval()
        model.likelihood.eval()
        with gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False, solves=False):
            with torch.no_grad():
                return model(torch.tensor(X)).mean.detach().numpy()

    elif "LaplacianModel" in type(model).__name__:
        return model.predict(X).reshape(-1)

    elif "interpolate" in module:
        return model(X)

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
def kernel_lengthscale_whatabouts(jsonfile, select_dimension):

    # Get either a false or a value
    lens = float(jsonfile['GPR_setup']['kernel_lengthscale'])
    if_ARD = jsonfile['GPR_setup']['if_kernel_lengthscale_ARD']

    if select_dimension == '3D':
        Ndim = 3
    elif select_dimension == '2D':
        Ndim = 2

    if if_ARD: lens = np.full(Ndim, lens)

    return lens




## Update kernel parameters
def update_kernel_params(model, new_lengthscale, new_variance=None):
    module = type(model).__module__

    #_ scikit-learn
    if "sklearn" in module:
        kernel = model.kernel
        params = {}
        klist  = kernel.get_params()

        for name in klist:
            if name.endswith('length_scale'):
                params[name] = new_lengthscale
            if new_variance and name.endswith('constant_value'):
                params[name] = new_variance

        kernel.set_params(**params)
        model.fit(model.X_train_, model.y_train_)
        return

    #_ GPflow
    elif "gpflow" in module:
        model.kernel.lengthscales.assign(new_lengthscale)
        if new_variance: model.kernel.variance.assign(new_variance)
        return

    #_ GPyTorch
    elif "gpytorch" in module:
        model.set_hyperparameters(new_lengthscale, new_variance)
        return




## NN trunk constructor
def build_nn_trunk(input_tensor, nn_layers):
    x = input_tensor
    for nn in nn_layers:
        x = layers.Dense(nn, activation=LeakyELU(beta=0.4), kernel_initializer=keras.initializers.GlorotUniform(seed=None))(x)
    return layers.Dense(1)(x)

def make_build_model(input_shape):
    def build_nn_trunk_tuner(hp):
        x = tf.keras.Input(shape=input_shape)

        nn_layers = hp.Int('num_layers', min_value=5, max_value=9)

        for i in range(nn_layers):
            units = hp.Int(f"units_{i}", min_value=60, max_value=480, step=60)
            x = layers.Dense(units, kernel_initializer=keras.initializers.GlorotUniform(seed=None))(x)
            x = LeakyELU()(x)

        return layers.Dense(1)(x)
    return build_nn_trunk_tuner



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
