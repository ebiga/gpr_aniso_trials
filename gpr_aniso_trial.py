import copy
import os
import hjson
import numpy as np
import pandas as pd
import sklearn
import silence_tensorflow.auto
import gpflow
import pickle
import time
import torch
import gpytorch
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import gpflow.utilities as gputil
import tensorflow as tf
import tensorflow_probability as tfp
import seaborn as sns

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
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


# Compute the rms of the mean
def check_mean(atest, mean, refd):
    delta = refd - mean

    Ntotal = np.shape(delta)[0]

    rms_check = np.sqrt( np.sum( delta**2. )/Ntotal )
    mae_check = np.sum( np.abs(delta) )/Ntotal
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
if_fold = False

GRID_POINTS = 5
NUM_REPEATS = 6
NUM_KERNELS = 2

variance_grid = np.exp(np.linspace(np.log(3.0), np.log(9.0), GRID_POINTS))
lengthss_grid = np.exp(np.linspace(np.log(4.0), np.log(8.0), GRID_POINTS))
scalings_grid = np.exp(np.linspace(np.log(0.05), np.log(0.5), GRID_POINTS))

stddev = 0.1

def random_search_gpflow_ard(datas, dataf, k=5, n_trials=NUM_REPEATS, n_jobs=4):
    # Prepare the infrastructure
    opt = gpflow.optimizers.Scipy()
    kf = KFold(n_splits=k)


    # A function to run a single combination of the hyperparameter grids
    #_ Option to run a KFold cross validation or direct "grid search"
    def evaluate_trial(trial_idx):
        rng = np.random.default_rng(42 + trial_idx)

        # Define the kernel parameters
        vars = rng.choice(variance_grid)
        lens = rng.choice(lengthss_grid)

        kernel = gpflow.kernels.RationalQuadratic(alpha=0.005, variance=vars, lengthscales=lens)
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.math.log(gpflow.utilities.to_default_float(vars)), stddev
        )
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(gpflow.utilities.to_default_float(lens)), stddev
        )
        gpflow.set_trainable(kernel.alpha, False)

        for otherks in range(NUM_KERNELS-1):
            facs = rng.choice(scalings_grid)
            vars = facs**2 * vars
            lens = facs * lens

            kkernel = gpflow.kernels.RationalQuadratic(alpha=0.005, variance=vars, lengthscales=lens)
            kkernel.variance.prior = tfp.distributions.LogNormal(
                tf.math.log(gpflow.utilities.to_default_float(vars)), stddev
            )
            kkernel.lengthscales.prior = tfp.distributions.LogNormal(
                tf.math.log(gpflow.utilities.to_default_float(lens)), stddev
            )
            gpflow.set_trainable(kkernel.alpha, False)

            kernel = kernel + kkernel

        # Save the init kernel as deep copy otherwise tf will overwrite
        kernelinit = copy.deepcopy(kernel)

        # zero out the best selection
        best_loss = float("inf")


        # Now get the thing done.
        if if_fold:
            # Fold you, fold me
            for train_index, val_index in kf.split(datas):
                # Retrive the initial kernel otherwise it'll restart
                kernel = copy.deepcopy(kernelinit)

                # Get the fold
                X_train, X_val = datas.iloc[train_index].to_numpy(), datas.iloc[val_index].to_numpy()
                y_train, y_val = dataf.iloc[train_index].to_numpy().reshape(-1, 1), dataf.iloc[val_index].to_numpy().reshape(-1, 1)

                # Create the full GPR model
                model = gpflow.models.GPR(data=(X_train, y_train), kernel=kernel, noise_variance=None)
                model.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
                gpflow.set_trainable(model.likelihood.variance, False)

                # Optimize the full model
                opt.minimize(model.training_loss, variables=model.trainable_variables, options=gpflow_options, compile=True)

                # Predict and compute loss
                y_pred, _ = model.posterior().predict_f(X_val)
                loss = np.mean((y_val - y_pred.numpy())**2)

                if loss < best_loss:
                    best_loss = loss

        else:
            # Optimize over the full dataset. This is a basic grid search method.
            model = gpflow.models.GPR(data=(datas.to_numpy(), dataf.to_numpy().reshape(-1,1)), kernel=kernel, noise_variance=None)
            model.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
            gpflow.set_trainable(model.likelihood.variance, False)

            opt.minimize(model.training_loss, variables=model.trainable_variables, options=gpflow_options, compile=True)

            best_loss = model.log_marginal_likelihood().numpy()


        print(f"Trial {trial_idx+1}/{n_trials}")
        print(f"  Kernel - init: {generate_gpflow_kernel_code(kernelinit)}")
        print(f"  Kernel - fine: {generate_gpflow_kernel_code(model.kernel)}")
        print(f"     Avg CV Loss: {best_loss:.6f}")

        return best_loss, kernelinit, model


    # Run trials in parallel
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(evaluate_trial)(i) for i in range(n_trials)
    )

    # Select the best model
    _, best_kernel, best_model = min(results, key=lambda x: x[0])
    print(f"  Kernel _full_ - init: {generate_gpflow_kernel_code(best_kernel)}")


    # Optimize over the full dataset
    if if_fold: 
        best_model = gpflow.models.GPR(data=(datas.to_numpy(), dataf.to_numpy().reshape(-1,1)), kernel=best_kernel, noise_variance=None)
        best_model.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
        gpflow.set_trainable(best_model.likelihood.variance, False)
        opt.minimize(best_model.training_loss, variables=best_model.trainable_variables, options=gpflow_options, compile=True)


    return best_model





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

# file locations
dafolder = method + "_" + select_dimension
os.makedirs(dafolder, exist_ok=True)

flightlog = open(os.path.join(dafolder, 'log.txt'), 'w')




### DATA POINTS

# training data
# in three different sizes
if select_input_size == 'full':
    data_base = pd.read_csv('./input_f.csv')

    NgridX = 141
    NgridY = 66
    NgridZ = 5

elif select_input_size == 'mid':
    data_base = pd.read_csv('./input_m.csv')

    NgridX = 36
    NgridY = 66
    NgridZ = 5

elif select_input_size == 'small':
    data_base = pd.read_csv('./input_s.csv')

    NgridX = 36
    NgridY = 33
    NgridZ = 5

elif select_input_size == 'tiny':
    data_base = pd.read_csv('./input_t.csv')

    NgridX = 18
    NgridY = 16
    NgridZ = 5

# test data
# extracted from the full case so only really meaningfull for the smaller cases
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

# separate the data sets into breakpoints and outputs
if select_dimension == '3D':
    filtin = data_base.index
elif select_dimension == '2D':
    filtin = data_base.loc[data_base['param3'] == param3fix].index

dataso = data_base.loc[filtin][brkpts].astype(np.float64)
dataf  = data_base.loc[filtin][output].astype(np.float64)

if select_dimension == '3D':
    filtin = test_base.index
elif select_dimension == '2D':
    filtin = test_base.loc[test_base['param3'] == param3fix].index

testso = test_base.loc[filtin][brkpts].astype(np.float64)
testf  = test_base.loc[filtin][output].astype(np.float64)

# make the breakpoints nondimensional, in the range [-0.5, 0.5]
NormMin = np.full(Ndimensions, 0.)
NormDlt = np.full(Ndimensions, 1.)

datas = dataso.copy()
tests = testso.copy()

for i, b in enumerate(brkpts):
    NormMini   = np.min(dataso[b])
    NormDlt[i] = np.max(dataso[b]) - NormMini
    NormMin[i] = NormMini/NormDlt[i] + 0.5

    datas[b] = dataso[b]/NormDlt[i] - NormMin[i]
    tests[b] = testso[b]/NormDlt[i] - NormMin[i]




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

        model = random_search_gpflow_ard(datas, dataf, k=5, n_jobs=6)

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
    plt.plot(np.array(loss), label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log(Loss)')
    plt.title('Loss Convergence')
    plt.legend()
    plt.savefig(os.path.join(dafolder, 'convergence_'+str(method)+'.png'))
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
    labels = ['ref', 'fitted']
    plt.legend(lines, labels)

    ax.set_xlabel('param1')
    ax.set_ylabel('param2')

    if select_dimension == '3D':
        pts_in_the_plot = param1_param2_cases[k]
    else:
        pts_in_the_plot = param1_param2_cases
    for c in param1_param2_cases:
        plt.scatter(c[1], c[2], lw=1, marker='x', label=c[0])
        plt.text(c[1], c[2], c[0], fontsize=9, ha='right', va='bottom')
        plt.plot([c[1], c[1]], [min(dataso['param2']), max(dataso['param2'])], 'k--', lw=0.25)

    plt.savefig(os.path.join(dafolder, 'the_contours_for_param3-'+str(v)+'.png'))
    plt.close()


# X-Ys
if select_dimension == '3D':
    params_to_range = ['param3', 'param2']
elif select_dimension == '2D':
    params_to_range = ['param2']

for c in param1_param2_cases:

    c_name = c[0]

    for pranged in params_to_range:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)

        if pranged == 'param3':
            psearch = 'param2'
            cx = c[2]
        elif pranged == 'param2':
            psearch = 'param3'
            cx = c[3]

        # get the closest points from the original "dimensional" data
        df = pd.DataFrame(dataso)
        if select_dimension == '3D':
            df['distance'] = np.sqrt((df['param1'] - c[1])**2 + (df[psearch] - cx)**2)
        else:
            df['distance'] = np.abs(df['param1'] - c[1])
        closest_points_index = df.loc[df['distance'] == df['distance'].min()].index

        param_range = np.linspace( min(dataso[pranged]), max(dataso[pranged]), 333 )

        # get the scattered points closest to the references
        XR = dataso.loc[closest_points_index][pranged]
        FR = dataf[closest_points_index]

        # Fit the data to generate the plot
        c_param1 = np.unique( df.loc[df['distance'] == df['distance'].min()]['param1'] ).item()
        if select_dimension == '3D':
            c_paramx = np.unique( df.loc[df['distance'] == df['distance'].min()][psearch] ).item()
            fig.suptitle("Condition  "+str(c_name)+": param1 "+str(round(c_param1,3))+"; " + psearch + " "+str(round(c_paramx,3)), fontsize=14)
        else:
            fig.suptitle("Condition  "+str(c_name)+": param1 "+str(round(c_param1,3)), fontsize=14)

        # create the X dimension to be fitted
        Xo = pd.DataFrame( {col: [pd.NA] * len(param_range) for col in datas.columns} )
        Xo['param1'] = c_param1
        if select_dimension == '3D': Xo[psearch] = c_paramx
        Xo[pranged] = param_range

        X = Xo.copy()
        for i, b in enumerate(brkpts):
            X[b] = Xo[b]/NormDlt[i] - NormMin[i]

        Y1 = my_predicts(model, X.to_numpy())

        # plot
        plt.plot(param_range, Y1.T, lw=0.5, label='fitted')
        plt.scatter(XR, FR.T, lw=0.5, marker='o', label='ref')

        ax.set_xlabel(pranged)
        ax.set_ylabel('var1')

        plt.legend()

        plt.savefig(os.path.join(dafolder, 'the_plot_for_'+str(c_name)+'_vs_'+pranged+'.png'))
        plt.close()


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
        color_index = int(4 * (testso.at[i, 'param3'] - param3_min)/param3_dlt)
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

plt.savefig(os.path.join(dafolder, 'one-to-one_for_'+str(c_name)+'.png'))
plt.close()


msg = f"Elapsed time: {time.time() - start_time:.2f} seconds"
print(msg)
flightlog.write(msg+'\n')
