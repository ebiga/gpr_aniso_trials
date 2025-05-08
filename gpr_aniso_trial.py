import os
import hjson
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

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['axes.titlesize']  = 16
matplotlib.rcParams['axes.labelsize']  = 16
matplotlib.rcParams['legend.fontsize'] = 14

import numpy as np
import pandas as pd
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

# get my functions
from auxfunctions import *
from auxgpytorch  import *
from optimme      import *
from diffusionme  import *

# set floats and randoms
gpflow.config.set_default_float('float64')
tf.keras.backend.set_floatx('float64')
torch.set_default_dtype(torch.float64)

tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
torch.manual_seed(42)





### IMPORTANT SHIZ

## FUNCTION: Just creates a model filename
def model_filename(method, dafolder):
    if method == 'gpr.scikit':
        trained_model_file = os.path.join(dafolder, 'model_training_' + method + '.pkl')
    elif method == 'gpr.gpflow':
        trained_model_file = os.path.join(dafolder, 'model_training_' + method + '.pkl')
    elif method == 'gpr.gpytorch':
        trained_model_file = os.path.join(dafolder, 'model_training_' + method + '.pth')
    elif method == 'nn.dense':
        trained_model_file = os.path.join(dafolder, 'model_training_' + method + '.keras')
    elif method == 'nn.attention':
        trained_model_file = os.path.join(dafolder, 'model_training_' + method + '.keras')

    return trained_model_file


## FUNCTION: Just loads a model file
def load_model_from_file(method, trained_model_file):
    likelihood = None

    if '.pkl' in trained_model_file:
        with open(trained_model_file, "rb") as f:
            model = pickle.load(f)

    elif '.keras' in trained_model_file:
        model = tf.keras.models.load_model(trained_model_file)

    elif method == 'gpr.gpytorch':
        model, likelihood = torch.load(trained_model_file, weights_only=False)

    return model, likelihood


## FUNCTION: Setup the model to be run and the file to save it
def get_me_a_model(method, DATAX, DATAF):

    ## Input model parameters
    #_ Define kernel parameters for GPRs
    if 'gpr' in method:
        #_ variance
        vars, if_train_variance = kernel_variance_whatabouts(casesetup)

        #_ lengthscale
        lens = kernel_lengthscale_whatabouts(casesetup, select_dimension)

    #_ Define nn parameters for... well, NNs
    if 'nn' in method:
        nn_layers = casesetup['keras_setup']["hidden_layers"]


    ## Each method has its own ways
    #_ GPR: Scikit-learn
    if method == 'gpr.scikit':
        # Set priors: fix what we don't want to optimise
        if not if_train_variance:
            kvars = ConstantKernel(vars, constant_value_bounds="fixed")
        else:
            kvars = ConstantKernel(vars)

        # Get a kernel
        kernel = kvars * RBF(length_scale=lens)

        # Setup the model
        #_ We set it here with no optimizer to dry start it - this will be changed later if needed
        n_restarts_optimizer = casesetup['GPR_setup']['scikit_setup']['n_restarts_optimizer']
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, optimizer=None)

        # We need to blind fit to create the kernel_ structure
        model.fit(DATAX, DATAF)

        return model, None

    #_ GPR: GPFlow
    elif method == 'gpr.gpflow':
        # Get a kernel
        kernel = gpflow.kernels.SquaredExponential(variance=vars, lengthscales=lens)

        # Set priors: fix what we don't want to optimise
        if not if_train_variance: gpflow.utilities.set_trainable(kernel.variance, False)

        # Set priors so the optimisation won't go wild
        pvar = 0.3
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(gpflow.utilities.to_default_float(lens)), pvar
        )

        # Create the full GPR model
        model = gpflow.models.GPR(data=(DATAX, DATAF.reshape(-1,1)), kernel=kernel, noise_variance=None)

        # GPflow requires us to build a likelihood, we want it noiseless and not trainable
        model.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
        gpflow.set_trainable(model.likelihood.variance, False)

        return model, model.likelihood

    #_ GPR: GPYTorch
    elif method == 'gpr.gpytorch':
        train_x = torch.tensor(DATAX)
        train_y = torch.tensor(DATAF)

        # Set priors: fix what we don't want to optimise
        if not if_train_variance: print('uh....')

        # Define the model
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.tensor(np.full(len(train_y),1.e-6)))
        model = GPRegressionModel(vars, lens, train_x, train_y, likelihood)

        # set the mode to training
        model.train()
        likelihood.train()

        return model, likelihood

    #_ NN, Dense
    elif method == 'nn.dense':
        # Setup the neural network
        input_shape = DATAX.shape[1:]

        model = keras.Sequential(
            [layers.Input(shape=input_shape)] +
            [layers.Dense(nn, activation='elu', kernel_initializer='he_normal') for nn in nn_layers] +
            [layers.Dense(1)]
            )

        return model, None

    #_ NN with Attention
    elif method == 'nn.attention':
        # Setup the neural network
        input_shape = DATAX.shape[1:]
        inputs = layers.Input(shape=input_shape)

        # Apply Multi-Head Attention
        re_inputs = ExpandALayer()(inputs)

        num_heads = casesetup['keras_setup']["multiheadattention_setup"]["num_heads"]
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=Ndimensions)(re_inputs, re_inputs)

        output = SqueezeALayer()(attention_output)

        # Fully connected layers
        for nn in nn_layers:
            output = layers.Dense(nn, activation='elu', kernel_initializer='he_normal')(output)
        output = layers.Dense(1)(output)

        # Create model
        model = keras.models.Model(inputs=inputs, outputs=output)

        return model, None





### USER OPTIONS

start_time = time.time()

with open('./casesetup.hjson', 'r') as casesetupfile:
    casesetup = hjson.load(casesetupfile)

select_dimension = casesetup['select_dimension']
select_input_size = casesetup['select_input_size']
method = casesetup['method']
if_train_optim = casesetup['if_train_optim']
figformat = casesetup['fig_format']

# file locations
dafolder = method + "_" + select_dimension + "_" + select_input_size
os.makedirs(dafolder, exist_ok=True)

flightlog = open(os.path.join(dafolder, 'log.txt'), 'w')





### DATA POINTS

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
    filtin = data_base.index
elif select_dimension == '2D':
    filtin = data_base.loc[(data_base['param3'] == param3fix)].index

dataso = data_base.loc[filtin][brkpts].astype(np.float64)
dataf  = data_base.loc[filtin][output].astype(np.float64)

if select_dimension == '3D':
    filtin = test_base.index
elif select_dimension == '2D':
    filtin = test_base.loc[(test_base['param3'] == param3fix)].index

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

XXo = np.unique( np.round(dataso['param1'], decimals=6) )
Dgrid[0] = XXo[1] - XXo[0]

YYo = np.unique( np.round(dataso['param2'], decimals=6) )
Dgrid[1] = YYo[1] - YYo[0]

if select_dimension == '3D':
    ZZo = np.unique( np.round(dataso['param3'], decimals=6) )
    Dgrid[2] = ZZo[1] - ZZo[0]

#_ nondimensionalisation
NormMin = np.full(Ndimensions, 0.)
NormDlt = np.full(Ndimensions, 1.)

for i, b in enumerate(brkpts):
    datalocl = dataso[b]
    NormMini = datalocl[np.argmin(np.abs(datalocl - 0.5 * (np.max(datalocl) + np.min(datalocl))))]

    NormDlt[i] = Dgrid[i]
    NormMin[i] = NormMini/NormDlt[i]

    datas[b] = dataso[b]/NormDlt[i] - NormMin[i]
    tests[b] = testso[b]/NormDlt[i] - NormMin[i]


# We need a vertex centered grid to evaluate the rms
if select_dimension == '3D':

    XX = XXo/NormDlt[0] - NormMin[0]
    YY = YYo/NormDlt[1] - NormMin[1]
    ZZ = ZZo/NormDlt[2] - NormMin[2]

    M_i, M_j, M_k = np.meshgrid(XX, YY, ZZ, indexing='ij')

    # staggered mesh
    staggeredpts_i = ( M_i[:-1, :-1, :-1] + M_i[1:, :-1, :-1] + M_i[:-1, 1:, :-1] + M_i[1:, 1:, :-1]
                     + M_i[:-1, :-1,1:  ] + M_i[1:, :-1,1:  ] + M_i[:-1, 1:,1:  ] + M_i[1:, 1:,1:  ]) / 8.
    staggeredpts_j = ( M_j[:-1, :-1, :-1] + M_j[1:, :-1, :-1] + M_j[:-1, 1:, :-1] + M_j[1:, 1:, :-1]
                     + M_j[:-1, :-1,1:  ] + M_j[1:, :-1,1:  ] + M_j[:-1, 1:,1:  ] + M_j[1:, 1:,1:  ]) / 8.
    staggeredpts_k = ( M_k[:-1, :-1, :-1] + M_k[1:, :-1, :-1] + M_k[:-1, 1:, :-1] + M_k[1:, 1:, :-1]
                     + M_k[:-1, :-1,1:  ] + M_k[1:, :-1,1:  ] + M_k[:-1, 1:,1:  ] + M_k[1:, 1:,1:  ]) / 8.

    staggeredpts = np.c_[staggeredpts_i.ravel(), staggeredpts_j.ravel(), staggeredpts_k.ravel()]

elif select_dimension == '2D':

    XX = XXo/NormDlt[0] - NormMin[0]
    YY = YYo/NormDlt[1] - NormMin[1]

    M_i, M_j = np.meshgrid(XX, YY, indexing='ij')

    # staggered mesh
    staggeredpts_i = ( M_i[:-1, :-1] + M_i[1:, :-1] + M_i[:-1, 1:] + M_i[1:, 1:] ) / 4.
    staggeredpts_j = ( M_j[:-1, :-1] + M_j[1:, :-1] + M_j[:-1, 1:] + M_j[1:, 1:] ) / 4.

    staggeredpts = np.c_[staggeredpts_i.ravel(), staggeredpts_j.ravel()]


# We'll need the shapes for managing in and out of the IJ meshgrid in the reversed order
shape_train_mesh = M_i.shape[::-1]
shape_stagg_mesh = np.shape(staggeredpts_i)

# Store the reference Laplacian metric
DDD = reshape_flatarray_like_reference_meshgrid(dataf.to_numpy(), shape_train_mesh, select_dimension)
laplacian_dataf = compute_Laplacian(DDD, DDD, select_dimension)





### TRAIN THE MODELS

trained_model_file = model_filename(method, dafolder)


# We need to build a model first
if if_train_optim == 'restart':
    # Read in a saved model from trained_model_file
    loss = None
    model, likelihood = load_model_from_file(method, trained_model_file)
else:
    # Otherwise build a model from scratch to be abused by the optimisers
    model, likelihood = get_me_a_model(method, datas.to_numpy(), dataf.to_numpy())


# Now we decide what to do with it
if if_train_optim == 'conventional':
    #_ Conventional optimisations
    loss = []
    if 'gpr' in method:
        model, likelihood = minimise_GPR_LML(method, model, likelihood, datas.to_numpy(), dataf.to_numpy(),
                                             trained_model_file, loss, casesetup, flightlog)
    elif 'nn' in method:
        minimise_NN_RMSE(method, model, likelihood, datas.to_numpy(), dataf.to_numpy(),
                         trained_model_file, loss, casesetup, flightlog)

elif if_train_optim == 'diffusionloss':
    #_ My diffusion loss optimisation
    loss = []
    if 'gpr' in method:
        model, likelihood = minimise_training_laplacian(model, datas.to_numpy(), dataf.to_numpy(), laplacian_dataf, staggeredpts,
                                                        select_dimension, shape_train_mesh, shape_stagg_mesh, loss,
                                                        casesetup, flightlog)

elif if_train_optim == 'nahimgood':
    #_ Just dry run
    print('Nothing here to run dry yet.')
    exit()


# Predict and evaluate
meanf = my_predicts(model, datas.to_numpy())
meant = my_predicts(model, tests.to_numpy())
flightlog.write(check_mean("Training", meanf, dataf.to_numpy()))
flightlog.write(check_mean("Testing", meant, testf.to_numpy()))
write_predicts_file(dafolder, testso, testf, meant)




### PLOTTING

# training convergence
if if_train_optim:
    plt.plot(np.array(loss), label='Training Loss')
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
    Z1 = dataf.loc[filtered_indices].to_numpy().reshape(Ngrid[1], Ngrid[0])

    COU = plt.contour(XXo, YYo, Z1, levels=levels, linestyles='solid', linewidths=1)

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
    XX, YY = np.meshgrid(XXo, YYo)

    Z1 = dataf.loc[filtered_indices].to_numpy().reshape(Ngrid[1], Ngrid[0])

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
predf_mesh = reshape_flatarray_like_reference_meshgrid(predf, shape_train_mesh, select_dimension)

predf_staggered = my_predicts(model, staggeredpts)
predf_staggeredmesh = predf_staggered.reshape(shape_stagg_mesh)

laplacian_predf = compute_Laplacian(predf_mesh, predf_staggeredmesh, select_dimension)

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
        k = np.where(np.isclose(ZZo, v, atol=1e-6))[0][0] - 1 # subtracting 1 cause the laplacians only have interior points

    XX, YY = np.meshgrid(XXo[1:-1], YYo[1:-1], indexing='ij')

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

    # __ Plot X-Ys
    fig, ax = plt.subplots(figsize=(12, 10))

    # X-direction slice at param1 = 14
    param1_val = 14
    idx_param1 = np.argmin(np.abs(XXo[1:-1] - param1_val))
    x_vals = YYo[1:-1]
    z1_x = np.log10(Z1[idx_param1, :])
    z2_x = np.log10(Z2[idx_param1, :])
    delta_x = np.abs(z1_x - z2_x)

    ax.plot(x_vals, z1_x, label='Reference', color='black', linewidth=1)
    ax.plot(x_vals, z2_x, label='Fitted', color='red', linestyle='--', linewidth=1)
    ax.plot(x_vals, delta_x, label='|Δ|', color='blue', linestyle=':', linewidth=1)
    ax.set_xlabel('param2')
    ax.set_ylabel('Log mod. Diffusion Operator')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 11)
    ax.set_ylim(-2.5, 0.5)
    ax.margins(0, tight=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(dafolder, f"Laplacian_param1-14_param3-{v}.{figformat}"), format=figformat, dpi=1200)
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
