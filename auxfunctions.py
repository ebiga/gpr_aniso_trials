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

gpflow.config.set_default_float('float64')
tf.keras.backend.set_floatx('float64')
torch.set_default_dtype(torch.float64)

tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
torch.manual_seed(42)




## Function to write out gpflow kernel params for the future
def generate_gpflow_kernel_code(kernel):
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
        #_ lenghtscales
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
    return kernel_to_code(kernel)




## Reshape due to csv XY and my lovely IJ orders
def reshape_flatarray_like_reference_meshgrid(offending_array, ashape, select_dimension):
    # the csv comes in the reversed order of the IJ mesh grid
    # the flattened array is reshaped into its mesh shape than tranposed to the IJ shape
    if select_dimension == '3D':
        return offending_array.reshape(ashape).transpose(2, 1, 0)
    elif select_dimension == '2D':
        return offending_array.reshape(ashape).transpose()




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
    
    else:
        if isinstance(model, gpytorch.models.GP):
            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                return model(torch.tensor(X)).mean.detach().numpy()
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
