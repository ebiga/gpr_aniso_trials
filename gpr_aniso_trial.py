import os
import hjson
import numpy as np
import pandas as pd
import scipy.optimize
import sklearn
import silence_tensorflow.auto
import gpflow
import pickle
import time
import torch
import gpytorch
import matplotlib
matplotlib.use('TkAgg')
matplotlib.set_loglevel('critical')

import matplotlib.pyplot as plt
import gpflow.utilities as gputil
import tensorflow as tf
import tensorflow_probability as tfp
import seaborn as sns

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from matplotlib import cm
from matplotlib.lines import Line2D
from tensorflow import keras
from keras import layers, saving
from gpflow.monitor import Monitor, MonitorTaskGroup
from sklearn.model_selection import ShuffleSplit, KFold
from joblib import Parallel, delayed

gpflow.config.set_default_float('float64')
tf.keras.backend.set_floatx('float64')
torch.set_default_dtype(torch.float64)

tf.random.set_seed(42)
keras.utils.set_random_seed(42)
torch.manual_seed(42)



# Function to write out gpflow kernel params for the future
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


# Dataset reduction for initial condition determination
def reduce_point_cloud(X, Y, target_fraction=0.5):
    """
    - X: Input dataset (numpy array, shape [N, D]).
    - Y: Target values (numpy array, shape [N, 1]).
    - target_fraction: Desired fraction of the total dataset to retain.
    """
    num_points, num_dims = X.shape

    # Compute the range of values for each dimension
    # Sort dimensions by range (descending)
    ranges = [np.ptp(X[:, i]) for i in range(num_dims)]
    sorted_dims = np.argsort(ranges)[::-1]

    # Determine the per-dimension reduction factor
    # Keep the smallest dimension intact
    dimensions_to_reduce = sorted_dims[:-1]
    per_dim_reduction = target_fraction ** (1 / len(dimensions_to_reduce))

    # Apply reduction logic
    mask = np.ones(num_points, dtype=bool)
    for dim in dimensions_to_reduce:
        unique_vals = np.unique( np.round(X[:, dim], decimals=6) )
        reduced_count = max(1, int(len(unique_vals) * per_dim_reduction))
        reduced_vals = unique_vals[:: len(unique_vals) // reduced_count][:reduced_count]
        mask &= np.isin(X[:, dim], reduced_vals)

    # Apply mask to reduce the dataset
    X_reduced = X[mask]
    Y_reduced = Y[mask]

    return X_reduced, Y_reduced


# Reshape due to csv XY and my lovely IJ orders
def reshape_flatarray_like_reference_meshgrid(offending_array, goodguy_meshgrid):
    # the csv comes in the reversed order of the IJ mesh grid
    reversed_shape = goodguy_meshgrid.shape[::-1]

    # the flattened array is reshaped into its mesh shape than tranposed to the IJ shape
    if select_dimension == '3D':
        return offending_array.reshape(reversed_shape).transpose(2, 1, 0)
    else:
        return offending_array.reshape(reversed_shape).transpose()


# Compute Laplacians for 2D and 3D, cell centred, for normal or staggered meshes
#_ The Laplacians are computed along the diagonals, so we can have only one staggered mesh
#_ It's never enough any effort to make life easier...
def compute_Laplacian(f_orig, f_stag):

    if f_orig is not f_stag:
        # If the arrays are not the same, we have a staggered mesh with the original mesh at the centre/corners
        #_ We compute the Laplacian with a 5-point stencil
        grid_spacing = 0.5

        if select_dimension == '3D':
            delta = 3. * grid_spacing**2.

            dsf_dD1s = np.abs(f_orig[2:  ,2:  ,2:  ] + f_orig[ :-2, :-2, :-2] - f_stag[1:  ,1:  ,1:  ] - f_stag[ :-1, :-1, :-1]) \
                           / (f_orig[2:  ,2:  ,2:  ] + f_orig[ :-2, :-2, :-2] + f_stag[1:  ,1:  ,1:  ] + f_stag[ :-1, :-1, :-1]) / (3.*delta)
            dsf_dD2s = np.abs(f_orig[ :-2,2:  ,2:  ] + f_orig[2:  , :-2, :-2] - f_stag[ :-1,1:  ,1:  ] - f_stag[1:  , :-1, :-1]) \
                           / (f_orig[ :-2,2:  ,2:  ] + f_orig[2:  , :-2, :-2] + f_stag[ :-1,1:  ,1:  ] + f_stag[1:  , :-1, :-1]) / (3.*delta)
            dsf_dD3s = np.abs(f_orig[2:  , :-2,2:  ] + f_orig[ :-2,2:  , :-2] - f_stag[1:  , :-1,1:  ] - f_stag[ :-1,1:  , :-1]) \
                           / (f_orig[2:  , :-2,2:  ] + f_orig[ :-2,2:  , :-2] + f_stag[1:  , :-1,1:  ] + f_stag[ :-1,1:  , :-1]) / (3.*delta)
            dsf_dD4s = np.abs(f_orig[2:  ,2:  , :-2] + f_orig[ :-2, :-2,2:  ] - f_stag[1:  ,1:  , :-1] - f_stag[ :-1, :-1,1:  ]) \
                           / (f_orig[2:  ,2:  , :-2] + f_orig[ :-2, :-2,2:  ] + f_stag[1:  ,1:  , :-1] + f_stag[ :-1, :-1,1:  ]) / (3.*delta)

            return dsf_dD1s + dsf_dD2s + dsf_dD3s + dsf_dD4s

        else:
            delta = 2. * grid_spacing**2.

            dsf_dD1s = np.abs(f_orig[2:  ,2:] + f_orig[ :-2,:-2] - f_stag[1:  ,1:] - f_stag[ :-1,:-1]) \
                           / (f_orig[2:  ,2:] + f_orig[ :-2,:-2] + f_stag[1:  ,1:] + f_stag[ :-1,:-1]) / (3.*delta)
            dsf_dD2s = np.abs(f_orig[ :-2,2:] + f_orig[2:  ,:-2] - f_stag[ :-1,1:] - f_stag[1:  ,:-1]) \
                           / (f_orig[ :-2,2:] + f_orig[2:  ,:-2] + f_stag[ :-1,1:] + f_stag[1:  ,:-1]) / (3.*delta)

            return dsf_dD1s + dsf_dD2s

    else:
        # This is the original training mesh processing, fstag = f_orig
        #_ We apply a 3-point stencil to compute the Laplacian
        grid_spacing = 1.

        if select_dimension == '3D':
            delta = 3. * grid_spacing**2.

            dsf_dD1s = np.abs(f_orig[2:  ,2:  ,2:  ] + f_orig[ :-2, :-2, :-2] - 2. * f_orig[1:-1,1:-1,1:-1]) \
                           / (f_orig[2:  ,2:  ,2:  ] + f_orig[ :-2, :-2, :-2] + 2. * f_orig[1:-1,1:-1,1:-1]) / delta
            dsf_dD2s = np.abs(f_orig[ :-2,2:  ,2:  ] + f_orig[2:  , :-2, :-2] - 2. * f_orig[1:-1,1:-1,1:-1]) \
                           / (f_orig[ :-2,2:  ,2:  ] + f_orig[2:  , :-2, :-2] + 2. * f_orig[1:-1,1:-1,1:-1]) / delta
            dsf_dD3s = np.abs(f_orig[2:  , :-2,2:  ] + f_orig[ :-2,2:  , :-2] - 2. * f_orig[1:-1,1:-1,1:-1]) \
                           / (f_orig[2:  , :-2,2:  ] + f_orig[ :-2,2:  , :-2] + 2. * f_orig[1:-1,1:-1,1:-1]) / delta
            dsf_dD4s = np.abs(f_orig[2:  ,2:  , :-2] + f_orig[ :-2, :-2,2:  ] - 2. * f_orig[1:-1,1:-1,1:-1]) \
                           / (f_orig[2:  ,2:  , :-2] + f_orig[ :-2, :-2,2:  ] + 2. * f_orig[1:-1,1:-1,1:-1]) / delta

            return dsf_dD1s + dsf_dD2s + dsf_dD3s + dsf_dD4s

        else:
            delta = 2. * grid_spacing**2.

            dsf_dD1s = np.abs(f_orig[2:  ,2:] + f_orig[ :-2,:-2] - 2. * f_orig[1:-1,1:-1]) \
                           / (f_orig[2:  ,2:] + f_orig[ :-2,:-2] + 2. * f_orig[1:-1,1:-1]) / delta
            dsf_dD2s = np.abs(f_orig[ :-2,2:] + f_orig[2:  ,:-2] - 2. * f_orig[1:-1,1:-1]) \
                           / (f_orig[ :-2,2:] + f_orig[2:  ,:-2] + 2. * f_orig[1:-1,1:-1]) / delta

            return dsf_dD1s + dsf_dD2s


# Write predicts out
def write_predicts_file(location, params_in, func_in, func_pred):
    predfile = pd.DataFrame(params_in)
    predfile['f'] = func_in
    predfile['f_pred'] = func_pred
    predfile.to_csv(os.path.join(location, 'test_data_out.csv'), index=False)


# Compute the rms of the mean
def check_mean(atest, mean, refd):
    delta = refd - mean

    rms_check = np.sqrt( np.mean( delta**2. ) )
    mae_check = np.mean( np.abs(delta) )
    max_check = np.max( np.abs(delta) )

    msg = atest + " errors: rms, mean, max: " + f"\t{rms_check:.3e};\t {mae_check:.3e};\t {max_check:.3e}\n"
    print(msg)
    return msg


# My wrapper of predict functions
def my_predicts(model, X):
    module = type(model).__module__
    
    if "sklearn" in module:
        return model.predict(X, return_cov=False)
    
    elif "gpflow" in module:
        return model.predict_f(X)[0].numpy().reshape(-1)
    
    elif "tensorflow" in module or "keras" in module:  # TensorFlow/Keras
        return model.predict(X).reshape(-1)
    
    else:
        if isinstance(model, gpytorch.models.GP):
            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                return model(torch.tensor(X)).mean.detach().numpy()
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")


# GPYTorch loves a class, doesn't it
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=Ndimensions, lengthscale=torch.tensor(np.full(Ndimensions, 1.0))), outputscale=1.0**2)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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


# A KFold thingy going on
NUM_KERNELS = 2
IF_ARD = True
bound = scipy.optimize.Bounds(0.005,500.)

alpha = 0.1

def get_me_a_kernel(alph, lens, vars=1.):
    lens = lens[0] if not IF_ARD else lens
    # tempk = gpflow.kernels.RationalQuadratic(alpha=alph, variance=vars, lengthscales=lens)
    # gpflow.set_trainable(tempk.alpha, False)
    # return tempk
    return gpflow.kernels.SquaredExponential(variance=vars, lengthscales=lens)

def minimise_training_laplacian(datas, dataf, histories):

    # A function to run a single combination of the hyperparameter grids
    #_ Option to run a KFold cross validation or direct "grid search"
    def evaluate_trial(x):

        # Define the kernel parameters
        lens = np.array([x[ilen] for ilen in range(1 + IF_ARD * (Ndimensions-1))])
        kernel = get_me_a_kernel(alpha, lens)

        for otherks in range(NUM_KERNELS-1):
            iref = (1 + IF_ARD * (Ndimensions-1))*(otherks+1)
            lens = np.array([x[ilen+iref] for ilen in range(1 + IF_ARD * (Ndimensions-1))])
            kernel = kernel + get_me_a_kernel(alpha, lens)

        # Optimize over the full dataset. This is a basic grid search method.
        model = gpflow.models.GPR(data=(datas.to_numpy(), dataf.to_numpy().reshape(-1,1)), kernel=kernel, noise_variance=None)
        model.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
        gpflow.set_trainable(model.likelihood.variance, False)

        # Estimate the loss metric at the staggered mesh
        #_ energy loss
        predf, _ = model.posterior().predict_f(datas.to_numpy())
        loss_e = np.sqrt(np.mean((predf.numpy().reshape(-1) - dataf.to_numpy())**2.))

        #_ momentum loss
        predf_mesh = reshape_flatarray_like_reference_meshgrid(predf.numpy(), XXX)

        predf_staggered, _ = model.posterior().predict_f(staggeredpts)
        predf_staggeredmesh = predf_staggered.numpy().reshape(np.shape(vertexmesh_X))

        laplacian_predf = compute_Laplacian(predf_mesh, predf_staggeredmesh)

        loss_m = np.sqrt(np.mean((laplacian_predf - laplacian_dataf)**2.))

        loss = loss_e + loss_m
        histories.append([np.log10(loss), np.log10(loss_e), np.log10(loss_m)])

        return loss


    # Optimizesss
    inits = np.array([0.5*np.exp(-k) for k in range(NUM_KERNELS) for count in range(1 + IF_ARD * (Ndimensions-1))])
    res = scipy.optimize.minimize(evaluate_trial, inits, method='COBYQA', bounds=bound, options=gpflow_options)

    # Assemble the final model
    lens = np.array([res.x[ilen] for ilen in range(1 + IF_ARD * (Ndimensions-1))])
    fkernel = get_me_a_kernel(alpha, lens)
    for otherks in range(NUM_KERNELS-1):
        iref = (1 + IF_ARD * (Ndimensions-1))*(otherks+1)
        lens = np.array([res.x[ilen+iref] for ilen in range(1 + IF_ARD * (Ndimensions-1))])
        fkernel = fkernel + get_me_a_kernel(alpha, lens)

    model = gpflow.models.GPR(data=(datas.to_numpy(), dataf.to_numpy().reshape(-1,1)), kernel=fkernel, noise_variance=None)
    model.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
    gpflow.set_trainable(model.likelihood.variance, False)

    return model





### USER OPTIONS
start_time = time.time()

with open('./casesetup.hjson', 'r') as casesetupfile:
    casesetup = hjson.load(casesetupfile)

select_dimension = casesetup['select_dimension']
select_input_size = casesetup['select_input_size']
method = casesetup['method']
if_train_optim = casesetup['if_train_optim']
r_numberofpoints = casesetup['gpflow_setup']['r_numberofpoints']
gpflow_options = casesetup['gpflow_setup']['optimiser']
keras_options = casesetup['keras_setup']
gpytorch_options = casesetup['gpytorch_setup']
n_restarts_optimizer = casesetup['scikit_setup']['n_restarts_optimizer']
figformat = casesetup['fig_format']

# file locations
dafolder = method + "_" + select_dimension + "_" + select_input_size
os.makedirs(dafolder, exist_ok=True)

flightlog = open(os.path.join(dafolder, 'log.txt'), 'w')




### DATA POINTS

if_negative_param2 = False

# TRAINING data
#_ in three different sizes
if select_input_size == 'full':
    data_base = pd.read_csv('./input_f.csv')
elif select_input_size == 'mid':
    data_base = pd.read_csv('./input_m.csv')
elif select_input_size == 'small':
    data_base = pd.read_csv('./input_s.csv')
elif select_input_size == 'tiny':
    data_base = pd.read_csv('./input_t.csv')

# TEST data
#_ extracted from the full case so only really meaningfull for the smaller cases
test_base = pd.read_csv('./test.csv')

# Define dimensions, breakpoints and output headers
if select_dimension == '3D':
    Ndimensions = 3 # first 3 columns have the breakpoints
elif select_dimension == '2D':
    Ndimensions = 2 # third column, param3, is downselected from the mid value
    param3fix = 0.7
else:
    print('ERROR: wrong "select_dimension".')
    exit()

brkpts = data_base.columns[:Ndimensions].to_numpy()
output = data_base.columns[-1]


# Separate the data sets into breakpoints and outputs
if select_dimension == '3D':
    if if_negative_param2:
        filtin = data_base.index
    else:
        filtin = data_base.loc[(data_base['param2'] >= 0)].index    
elif select_dimension == '2D':
    if if_negative_param2:
        filtin = data_base.loc[(data_base['param3'] == param3fix)].index
    else:
        filtin = data_base.loc[(data_base['param3'] == param3fix) & (data_base['param2'] >= 0)].index

dataso = data_base.loc[filtin][brkpts].astype(np.float64)
dataf  = data_base.loc[filtin][output].astype(np.float64)

if select_dimension == '3D':
    if if_negative_param2:
        filtin = test_base.index
    else:
        filtin = test_base.loc[(test_base['param2'] >= 0)].index    
elif select_dimension == '2D':
    if if_negative_param2:
        filtin = test_base.loc[(test_base['param3'] == param3fix)].index
    else:
        filtin = test_base.loc[(test_base['param3'] == param3fix) & (test_base['param2'] >= 0)].index

testso = test_base.loc[filtin][brkpts].astype(np.float64)
testf  = test_base.loc[filtin][output].astype(np.float64)


# Make the breakpoints nondimensional by unitary grid spacing
datas = dataso.copy()
tests = testso.copy()

#_ number of points
Ngrid = np.zeros(Ndimensions, dtype=int)
for n, b in enumerate(brkpts):
    Ngrid[n] = len(np.unique( np.round(dataso[b], decimals=6) ))

#_ mesh spacing
Dgrid = np.full(Ndimensions, 1.)
for n, b in enumerate(brkpts):
    Dgrid[n] = (np.max(dataso[b]) - np.min(dataso[b]))/(Ngrid[n] - 1)

#_ nondimensionalisation
NormMin = np.full(Ndimensions, 0.)
NormDlt = np.full(Ndimensions, 1.)

for i, b in enumerate(brkpts):
    NormMini   = 0.5*(np.max(dataso[b]) - np.min(dataso[b])) + np.min(dataso[b])
    NormDlt[i] = Dgrid[i]
    NormMin[i] = NormMini/NormDlt[i]

    datas[b] = dataso[b]/NormDlt[i] - NormMin[i]
    tests[b] = testso[b]/NormDlt[i] - NormMin[i]


# We need a vertex centered grid to evaluate the rms
if select_dimension == '3D':

    XX = np.unique( np.round(dataso['param1'], decimals=6) )/NormDlt[0] - NormMin[0]
    YY = np.unique( np.round(dataso['param2'], decimals=6) )/NormDlt[1] - NormMin[1]
    ZZ = np.unique( np.round(dataso['param3'], decimals=6) )/NormDlt[2] - NormMin[2]
    XXX, YYY, ZZZ = np.meshgrid(XX, YY, ZZ, indexing='ij')

    # staggered mesh
    vertexmesh_X = ( XXX[:-1, :-1, :-1] + XXX[1:, :-1, :-1] + XXX[:-1, 1:, :-1] + XXX[1:, 1:, :-1]
                   + XXX[:-1, :-1,1:  ] + XXX[1:, :-1,1:  ] + XXX[:-1, 1:,1:  ] + XXX[1:, 1:,1:  ]) / 8
    vertexmesh_Y = ( YYY[:-1, :-1, :-1] + YYY[1:, :-1, :-1] + YYY[:-1, 1:, :-1] + YYY[1:, 1:, :-1]
                   + YYY[:-1, :-1,1:  ] + YYY[1:, :-1,1:  ] + YYY[:-1, 1:,1:  ] + YYY[1:, 1:,1:  ]) / 8
    vertexmesh_Z = ( ZZZ[:-1, :-1, :-1] + ZZZ[1:, :-1, :-1] + ZZZ[:-1, 1:, :-1] + ZZZ[1:, 1:, :-1]
                   + ZZZ[:-1, :-1,1:  ] + ZZZ[1:, :-1,1:  ] + ZZZ[:-1, 1:,1:  ] + ZZZ[1:, 1:,1:  ]) / 8

    staggeredpts = np.c_[vertexmesh_X.ravel(), vertexmesh_Y.ravel(), vertexmesh_Z.ravel()]

elif select_dimension == '2D':

    XX = np.unique( np.round(dataso['param1'], decimals=6) )/NormDlt[0] - NormMin[0]
    YY = np.unique( np.round(dataso['param2'], decimals=6) )/NormDlt[1] - NormMin[1]
    XXX, YYY = np.meshgrid(XX, YY, indexing='ij')

    # staggered mesh
    vertexmesh_X = ( XXX[:-1, :-1] + XXX[1:, :-1] + XXX[:-1, 1:] + XXX[1:, 1:] ) / 4
    vertexmesh_Y = ( YYY[:-1, :-1] + YYY[1:, :-1] + YYY[:-1, 1:] + YYY[1:, 1:] ) / 4

    staggeredpts = np.c_[vertexmesh_X.ravel(), vertexmesh_Y.ravel()]

# Store the reference Laplacian metric
DDD = reshape_flatarray_like_reference_meshgrid(dataf.to_numpy(), XXX)
laplacian_dataf = compute_Laplacian(DDD, DDD)



### TRAIN THE MODELS

if method == 'gpr.scikit':
    loss = []
    trained_model_file = os.path.join(dafolder, 'model_training_' + method + '.pkl')

    if if_train_optim:
        # Define the kernel parameters
        kernel = 1.**2 * RBF(length_scale=np.full(Ndimensions, 1.0))

        # Setup the model
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)

        # Train model
        model.fit(datas.to_numpy(), dataf.to_numpy())

        msg = "Training Kernel: " + str(model.kernel_)
        print(msg)
        flightlog.write(msg+'\n')
 
        # store the model for reuse
        with open(trained_model_file, "wb") as f:
            pickle.dump(model, f)
    else:
        # We simply insert the input data into the kernel
        with open(trained_model_file, "rb") as f:
            model = pickle.load(f)
    
    # Predict and evaluate
    meanf = my_predicts(model, datas.to_numpy())
    meant = my_predicts(model, tests.to_numpy())
    flightlog.write(check_mean("Training", meanf, dataf.to_numpy()))
    flightlog.write(check_mean("Testing", meant, testf.to_numpy()))



elif method == 'gpr.gpflow':
    loss = []
    trained_model_file = os.path.join(dafolder, 'model_training_' + method + '.pkl')

    if if_train_optim:

        model = minimise_training_laplacian(datas, dataf, loss)

        msg = "Training Kernel: " + str(generate_gpflow_kernel_code(model.kernel))
        print(msg)
        flightlog.write(msg+'\n')

        # store the model for reuse
        with open(trained_model_file, "wb") as f:
            pickle.dump(model, f)

    else:
        # We simply insert the input data into the kernel
        with open(trained_model_file, "rb") as f:
            model = pickle.load(f)

    # store the posterior for faster prediction
    posterior_gpr = model.posterior()

    # Predict and evaluate
    meanf = my_predicts(posterior_gpr, datas.to_numpy())
    meant = my_predicts(posterior_gpr, tests.to_numpy())
    flightlog.write(check_mean("Training", meanf, dataf.to_numpy()))
    flightlog.write(check_mean("Testing", meant, testf.to_numpy()))
    write_predicts_file(dafolder, testso, testf, meant)



elif method == 'gpr.gpytorch':
    loss = []
    trained_model_file = os.path.join(dafolder, 'model_training_' + method + '.pth')

    if if_train_optim:
        # Convert data to torch tensors
        train_x = torch.tensor(datas.to_numpy())
        train_y = torch.tensor(dataf.to_numpy())

        # Define the model
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.tensor(np.full(len(train_y),1.e-6)))
        model = GPRegressionModel(train_x, train_y, likelihood)

        # set the mode to training
        model.train()
        likelihood.train()

        # Use Adam, hey Adam, me again, an apple
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(gpytorch_options['maxiter']):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            opt_loss = -mll(output, train_y)
            opt_loss.backward()
            loss.append(opt_loss.item())
            optimizer.step()

        msg = "Lengthscale: " + str(model.covar_module.base_kernel.lengthscale.squeeze().tolist()) + "\n" \
            + "Variance: " + str(model.covar_module.outputscale.item()) + "\n"

        print(msg)
        flightlog.write(msg)

        # store the model for reuse
        torch.save((model, likelihood), trained_model_file)

    else:
        # We simply insert the input data into the kernel
        model, likelihood = torch.load(trained_model_file, weights_only=False)

    # set the mode to eval
    model.eval()
    likelihood.eval()

    # Predict and evaluate
    meanf = my_predicts(model, datas.to_numpy())
    meant = my_predicts(model, tests.to_numpy())
    flightlog.write(check_mean("Training", meanf, dataf.to_numpy()))
    flightlog.write(check_mean("Testing", meant, testf.to_numpy()))



elif method == 'nn.tf':
    loss = []
    trained_model_file = os.path.join(dafolder, 'model_training_' + method + '.keras')

    if if_train_optim:
        input_shape = datas.to_numpy().shape[1:]

        # Setup the neural network
        model = keras.Sequential(
            [layers.Input(shape=input_shape)] +
            [layers.Dense(nn, activation='elu', kernel_initializer='he_normal') for nn in keras_options["hidden_layers"]] +
            [layers.Dense(1)]
            )

        model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(learning_rate=keras_options["learning_rate"]))

        history = model.fit(
            datas.to_numpy(),
            dataf.to_numpy(),
            verbose=0, epochs=keras_options["epochs"], batch_size=keras_options["batch_size"],
            )
        loss = np.log(history.history['loss'])

        # store the model for reuse
        model.save(trained_model_file)
    
    else:
        # We simply insert the input data into the kernel
        model = tf.keras.models.load_model(trained_model_file)

    # Predict and evaluate
    meanf = my_predicts(model, datas.to_numpy())
    meant = my_predicts(model, tests.to_numpy())
    flightlog.write(check_mean("Training", meanf, dataf.to_numpy()))
    flightlog.write(check_mean("Testing", meant, testf.to_numpy()))



elif method == 'at.tf':
    loss = []
    trained_model_file = os.path.join(dafolder, 'model_training_' + method + '.keras')

    if if_train_optim:
        input_shape = datas.to_numpy().shape[1:]
    
        # Setup the neural network
        inputs = layers.Input(shape=input_shape)

        # Apply Multi-Head Attention
        re_inputs = ExpandALayer()(inputs)
        attention_output = layers.MultiHeadAttention(num_heads=keras_options["multiheadattention_setup"]["num_heads"], key_dim=Ndimensions)(re_inputs, re_inputs)
        output = SqueezeALayer()(attention_output)

        # Fully connected layers
        for nn in keras_options["hidden_layers"]:
            output = layers.Dense(nn, activation='elu', kernel_initializer='he_normal')(output)
        output = layers.Dense(1)(output)

        # Create model
        model = keras.models.Model(inputs=inputs, outputs=output)

        model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(learning_rate=keras_options["learning_rate"]))

        history = model.fit(
            datas.to_numpy(),
            dataf.to_numpy(),
            verbose=0, epochs=keras_options["epochs"], batch_size=keras_options["batch_size"],
            )
        loss = np.log(history.history['loss'])

        # store the model for reuse
        model.save(trained_model_file)
    
    else:
        # We simply insert the input data into the kernel
        model = tf.keras.models.load_model(trained_model_file)

    # Predict and evaluate
    meanf = my_predicts(model, datas.to_numpy())
    meant = my_predicts(model, tests.to_numpy())
    flightlog.write(check_mean("Training", meanf, dataf.to_numpy()))
    flightlog.write(check_mean("Testing", meant, testf.to_numpy()))




### PLOTTING

# training convergence
if if_train_optim:
    plt.plot(np.array(loss)[:,0], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log(Loss)')
    plt.title('Loss Convergence')
    plt.legend()
    plt.savefig(os.path.join(dafolder, 'convergence_'+str(method)+'.'+figformat), format=figformat, dpi=1200)
    plt.close()


# reference points to plot, provided in the original "dimensional" space
param1_param2_cases = [['c1', 13.25, 1.39, 0.7], ['c2', 27.8, 7.4, 0.8]]

if select_dimension == '3D':
    param3_cases = [0.7, 0.8]
elif select_dimension == '2D':
    param3_cases = [select_dimension]

# contours
for k, v in enumerate(param3_cases):
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Param3 "+str(v), fontsize=14)
    ax = fig.add_subplot(111)

    # define the levels for the plot
    levels = np.arange(0.04,0.24,0.02)

    # prepare the arrays
    ngrid = 51

    So = pd.DataFrame( {col: [pd.NA] * ngrid*ngrid for col in dataso.columns} )

    X = np.linspace( min(dataso['param1']), max(dataso['param1']), ngrid )
    Y = np.linspace( min(dataso['param2']), max(dataso['param2']), ngrid )

    XX, YY = np.meshgrid(X, Y)

    So['param1'] = XX.ravel()
    So['param2'] = YY.ravel()

    if select_dimension == '3D':
        So['param3'] = v
        filtered_indices = dataso[ np.round(dataso['param3'], decimals=6) == v].index
    else:
        filtered_indices = dataso.index

    S = So.copy()
    for i, b in enumerate(brkpts):
        S[b] = So[b]/NormDlt[i] - NormMin[i]

    Z2 = my_predicts(model, S.to_numpy()).reshape(ngrid, ngrid)

    COF = plt.contour(X, Y, Z2, levels=levels, linestyles='dashed', linewidths=0.5)

    # fetch the reference data
    X = np.unique( np.round(dataso.loc[filtered_indices]['param1'], decimals=6) )
    Y = np.unique( np.round(dataso.loc[filtered_indices]['param2'], decimals=6) )

    Z1 = dataf.loc[filtered_indices].to_numpy().reshape(len(Y), len(X))

    COU = plt.contour(X, Y, Z1, levels=levels, linestyles='solid', linewidths=1)

    # finalise the plot
    plt.clabel(COU, fontsize=9)

    lines = [
        Line2D([0], [0], color='black', linestyle='solid' , linewidth=1.0),
        Line2D([0], [0], color='black', linestyle='dashed', linewidth=0.5),
    ]
    labels = ['Reference', 'Fitted']
    plt.legend(lines, labels)

    ax.set_xlabel('param1')
    ax.set_ylabel('param2')

    if select_dimension == '3D':
        pts_in_the_plot = [param1_param2_cases[k]]
    else:
        pts_in_the_plot = param1_param2_cases
    for c in pts_in_the_plot:
        plt.scatter(c[1], c[2], lw=1, marker='x', label=c[0])
        plt.text(c[1], c[2], c[0], fontsize=9, ha='right', va='bottom')
        plt.plot([c[1], c[1]], [min(dataso['param2']), max(dataso['param2'])], 'k--', lw=0.25)
        plt.plot([min(dataso['param1']), max(dataso['param1'])], [c[2], c[2]], 'k--', lw=0.25)

    plt.savefig(os.path.join(dafolder, 'the_contours_for_param3-'+str(v)+'.'+figformat), format=figformat, dpi=1200)
    plt.close()


# surfaces
for k, v in enumerate(param3_cases):
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Surface - param3 "+str(v), fontsize=14)
    ax = fig.add_subplot(projection='3d')

    # prepare the arrays
    ngrid = 250

    So = pd.DataFrame( {col: [pd.NA] * ngrid*ngrid for col in dataso.columns} )

    X = np.linspace( min(dataso['param1']), max(dataso['param1']), ngrid )
    Y = np.linspace( min(dataso['param2']), max(dataso['param2']), ngrid )

    XX, YY = np.meshgrid(X, Y)

    So['param1'] = XX.ravel()
    So['param2'] = YY.ravel()

    if select_dimension == '3D':
        So['param3'] = v
        filtered_indices = dataso[ np.round(dataso['param3'], decimals=6) == v].index
    else:
        filtered_indices = dataso.index

    S = So.copy()
    for i, b in enumerate(brkpts):
        S[b] = So[b]/NormDlt[i] - NormMin[i]

    Z2 = my_predicts(model, S.to_numpy()).reshape(ngrid, ngrid)

    ax.plot_surface(XX, YY, Z2, cmap=cm.seismic, linewidth=0, alpha=0.5, antialiased=True, label="Fitted")

    # fetch the reference data
    X = np.unique( np.round(dataso.loc[filtered_indices]['param1'], decimals=6) )
    Y = np.unique( np.round(dataso.loc[filtered_indices]['param2'], decimals=6) )

    XX, YY = np.meshgrid(X, Y)

    Z1 = dataf.loc[filtered_indices].to_numpy().reshape(len(Y), len(X))

    ax.plot_wireframe(XX, YY, Z1, color='black', linewidth=0.4, label="Reference")

    ax.view_init(elev=20, azim=135)
    ax.legend()

    ax.set_xlabel('param1')
    ax.set_ylabel('param2')

    plt.savefig(os.path.join(dafolder, 'the_surface_for_param3-'+str(v)+'.'+figformat), format=figformat, dpi=1200)
    plt.close()


# Laplacians
#_ Build the staggered mesh info to plot and write out the RMSE
predf = my_predicts(model, datas.to_numpy())
predf_mesh = reshape_flatarray_like_reference_meshgrid(predf, XXX)

predf_staggered = my_predicts(model, staggeredpts)
predf_staggeredmesh = predf_staggered.reshape(np.shape(vertexmesh_X))

laplacian_predf = compute_Laplacian(predf_mesh, predf_staggeredmesh)

loss_m = np.sqrt(np.mean((laplacian_predf - laplacian_dataf)**2.))
msg = f"RMSE of the Laplacians: {loss_m:.3e}"
print(msg)
flightlog.write(msg+'\n')

#_ Now to the plots
for k, v in enumerate(param3_cases):
    #__ Plot the surfaces
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Laplacian - param3 "+str(v), fontsize=14)
    ax = fig.add_subplot(projection='3d')

    if select_dimension == '3D':
        param3_vals = np.unique(np.round(dataso['param3'], decimals=6))
        k = np.where(np.isclose(param3_vals, v, atol=1e-6))[0][0] - 1 # subtracting 1 cause the laplacians only have interior points

    X = np.unique(np.round(dataso['param1'], decimals=6))
    Y = np.unique(np.round(dataso['param2'], decimals=6))
    XX, YY = np.meshgrid(X[1:-1], Y[1:-1], indexing='ij')

    # filtered data in the plane
    Z1 = laplacian_dataf[:, :, k] if laplacian_dataf.ndim == 3 else laplacian_dataf
    Z2 = laplacian_predf[:, :, k] if laplacian_predf.ndim == 3 else laplacian_predf

    # aight
    ax.plot_wireframe(XX, YY, Z1, color='black', linewidth=0.4, label="Reference")
    ax.plot_surface(XX, YY, Z2, cmap=cm.spring, linewidth=0.4, alpha=0.7, antialiased=False, shade=True, label="Fitted")

    ax.view_init(elev=20, azim=135)
    ax.legend()

    ax.set_xlabel('param1')
    ax.set_ylabel('param2')
    ax.set_zlabel('Laplacian(var1)')

    plt.savefig(os.path.join(dafolder, 'the_Laplacian_for_param3-'+str(v)+'.'+figformat), format=figformat, dpi=1200)
    plt.close()

    #__ Plot X-Ys
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Midline Laplacian Comparison (log scale) - param3 = {v}", fontsize=14)

    # X-direction slice (mid param1)
    mid_x = len(X[1:-1]) // 2
    x_vals = Y[1:-1]
    z1_x = np.log10(Z1[mid_x, :])
    z2_x = np.log10(Z2[mid_x, :])
    delta_x = np.abs(z1_x - z2_x)

    axs[0].plot(x_vals, z1_x, label='Reference', color='black', linewidth=1)
    axs[0].plot(x_vals, z2_x, label='Fitted', color='red', linestyle='--', linewidth=1)
    axs[0].plot(x_vals, delta_x, label='|Δ|', color='blue', linestyle=':', linewidth=1)
    axs[0].set_title('Slice along param2 (mid param1)')
    axs[0].set_xlabel('param2')
    axs[0].set_ylabel('Log Laplacian(var1)')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim(0,11)
    axs[0].set_ylim(-2.5,0.5)
    axs[0].margins(0, x=None, y=None, tight=True)

    # Y-direction slice (mid param2)
    mid_y = len(Y[1:-1]) // 2
    y_vals = X[1:-1]
    z1_y = np.log10(Z1[:, mid_y])
    z2_y = np.log10(Z2[:, mid_y])
    delta_y = np.abs(z1_y - z2_y)

    axs[1].plot(y_vals, z1_y, label='Reference', color='black', linewidth=1)
    axs[1].plot(y_vals, z2_y, label='Fitted', color='red', linestyle='--', linewidth=1)
    axs[1].plot(y_vals, delta_y, label='|Δ|', color='blue', linestyle=':', linewidth=1)
    axs[1].set_title('Slice along param1 (mid param2)')
    axs[1].set_xlabel('param1')
    axs[1].set_ylabel('Log Laplacian(var1)')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlim(0,35)
    axs[1].set_ylim(-2.5,0.5)
    axs[1].margins(0, x=None, y=None, tight=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(dafolder, f"Laplacian_crosssection_param3-{v}."+figformat), format=figformat, dpi=1200)
    plt.close()


# X-Ys
if select_dimension == '3D':
    params_to_range = ['param3', 'param2']
elif select_dimension == '2D':
    params_to_range = ['param2', 'param1']

for c in param1_param2_cases:

    c_name = c[0]

    for pranged in params_to_range:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)

        if select_dimension == '3D':
            # param1 is fixed
            if pranged == 'param3':
                psearch = 'param2'
                cx = c[2]
            elif pranged == 'param2':
                psearch = 'param3'
                cx = c[3]
        elif select_dimension == '2D':
            if pranged == 'param2':
                psearch = 'param1'
                cx = c[1]
            elif pranged == 'param1':
                psearch = 'param2'
                cx = c[2]

        # get the closest points from the original "dimensional" data
        df = pd.DataFrame(dataso)
        if select_dimension == '3D':
            df['distance'] = np.sqrt((df['param1'] - c[1])**2 + (df[psearch] - cx)**2)
        else:
            df['distance'] = np.abs(df[psearch] - cx)
        closest_points_index = df.loc[df['distance'] == df['distance'].min()].index

        param_range = np.linspace( min(dataso[pranged]), max(dataso[pranged]), 333 )

        # get the scattered points closest to the references
        XR = dataso.loc[closest_points_index][pranged]
        FR = dataf[closest_points_index]

        # Fit the data to generate the plot
        if select_dimension == '3D':
            c_param1 = np.unique( df.loc[df['distance'] == df['distance'].min()]['param1'] ).item()
            c_paramx = np.unique( df.loc[df['distance'] == df['distance'].min()][ psearch] ).item()
            fig.suptitle("Condition  "+str(c_name)+": param1 "+str(round(c_param1,3))+"; " + psearch + " "+str(round(c_paramx,3)), fontsize=14)
        else:
            c_paramx = np.unique( df.loc[df['distance'] == df['distance'].min()][ psearch] ).item()
            fig.suptitle("Condition  "+str(c_name)+": param1 "+str(round(c_paramx,3)), fontsize=14)

        # create the X dimension to be fitted
        Xo = pd.DataFrame( {col: [pd.NA] * len(param_range) for col in datas.columns} )
        if select_dimension == '3D': Xo['param1'] = c_param1
        Xo[psearch] = c_paramx
        Xo[pranged] = param_range

        X = Xo.copy()
        for i, b in enumerate(brkpts):
            X[b] = Xo[b]/NormDlt[i] - NormMin[i]

        Y1 = my_predicts(model, X.to_numpy())

        # plot
        plt.plot(param_range, Y1.T, lw=0.5, label='Fitted')
        plt.scatter(XR, FR.T, lw=0.5, marker='o', label='Reference')

        ax.set_xlabel(pranged)
        ax.set_ylabel('var1')

        plt.legend()

        plt.savefig(os.path.join(dafolder, 'the_plot_for_'+str(c_name)+'_vs_'+pranged+'.'+figformat), format=figformat, dpi=1200)
        plt.close()

        # save the dat file
        Xo['predf'] = Y1
        Xo.to_csv(os.path.join(dafolder, str(c_name)+'_vs_'+pranged+'.csv'), index=False)


# 1:1 expected vs. fitted
num_points = len(testf)

# Define symbols and sizes
markers = ['*', '^', 's', 'D']
sizes   = [30, 60, 90, 120]
colors  = sns.color_palette("husl", 5)

# Assign quartiles
param1_q = np.digitize(testso['param1'], np.percentile(testso['param1'], [25, 50, 75]), right=True)
param2_q = np.digitize(testso['param2'], np.percentile(testso['param2'], [25, 50, 75]), right=True)

if select_dimension == '3D':
    param3_max = max(testso['param3'])
    param3_min = min(testso['param3'])
    param3_dlt = param3_max - param3_min

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(num_points):

    if select_dimension == '3D':
        color_index = int(4 * (testso.iloc[i]['param3'] - param3_min)/param3_dlt)
    else:
        color_index = 0

    ax.scatter(
        testf.to_numpy()[i], meant[i],
        color=colors[color_index],
        marker=markers[param2_q[i]],
        s=sizes[param1_q[i]],
        alpha=0.75
    )

# 1:1 line
ax.plot([0, 1], [0, 1], 'k--')

# Legend for markers
legend_markers = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='black', markersize=10) for m in markers]
marker_legend = ax.legend(legend_markers, [f'Q{i+1} of Param2' for i in range(4)], title="Marker: Param2", loc="upper right")

# Legend for sizes
legend_sizes = [plt.scatter([], [], s=s, color='black') for s in sizes]
size_legend = ax.legend(legend_sizes, [f'Q{i+1} of Param1' for i in range(4)], title="Size: Param1", loc="upper left")

# Legend for colors
if select_dimension == '3D':
    legend_colors = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10) for c in colors]
    color_legend = ax.legend(legend_colors, [f'Param3 = {i}' for i in range(5)], title="Color: Param3", loc="lower right")

ax.add_artist(marker_legend)
ax.add_artist(size_legend)

ax.set_xlabel("Expected")
ax.set_ylabel("Fitted")
ax.set_title("Testing Space 1:1")

plt.savefig(os.path.join(dafolder, 'one-to-one_for_'+str(c_name)+'.'+figformat), format=figformat, dpi=1200)
plt.close()


msg = f"Elapsed time: {time.time() - start_time:.2f} seconds"
print(msg)
flightlog.write(msg+'\n')
