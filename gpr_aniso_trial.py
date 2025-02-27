import numpy as np
import pandas as pd
import sklearn
import silence_tensorflow.auto
import gpflow
import time
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import gpflow.utilities as gputil
import tensorflow as tf
import tensorflow_probability as tfp

tf.random.set_seed(42)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from matplotlib.lines import Line2D
from tensorflow import keras
from keras import layers

gpflow.config.set_default_float('float64')

# gpflow minimise options
options = {
    "maxiter": 1500,
    "gtol": 1e-9,
    "ftol": 1e-9,
}


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
        param_nam = "variance"
        param_val = k.variance.numpy()
        params.append(f"{param_nam}={param_val}")
        #_ lenghtscales
        param_nam = "lengthscales"
        param_val = k.lengthscales.numpy()
        params.append(f"{param_nam}={param_val}")
        # Construct kernel initialization code
        kernel_name = type(k).__name__
        return f"gpflow.kernels.{kernel_name}({', '.join(params)})"
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
        unique_vals = np.unique(X[:, dim])
        reduced_count = max(1, int(len(unique_vals) * per_dim_reduction))
        reduced_vals = unique_vals[:: len(unique_vals) // reduced_count][:reduced_count]
        mask &= np.isin(X[:, dim], reduced_vals)

    # Apply mask to reduce the dataset
    X_reduced = X[mask]
    Y_reduced = Y[mask]

    return X_reduced, Y_reduced


flightlog = open('log.txt', 'w')

start_time = time.time()

### user options
# options: 'scikit'   or 'gpflow'
method = 'nn.tf'
# options: ain't it faking obvious?...
if_train_optim = True
# booh!lean
if_train_aniso = True
# initial condition ratio of points
r_numberofpoints = 0.5



### data points

## training space
data_bases = pd.read_csv('./input.csv')

Ndimensions = 3 # first 3 columns have the breakpoints

brkpts = data_bases.columns[:Ndimensions].to_numpy()
output = data_bases.columns[Ndimensions]

# clean up too much data, like sampling at every 4 row
param1_steps = 4

data_basel = data_bases.iloc[::param1_steps]

# separate breakpoints and output
dataso = data_basel[brkpts].astype(np.float64)
dataf  = data_basel[output].astype(np.float64)

# make this nondimensional
NormMin = np.full(Ndimensions, 0.)
NormDlt = np.full(Ndimensions, 1.)
datas = dataso.copy()
for i, b in enumerate(brkpts):
    NormMini   = np.min(dataso[b])
    NormDlt[i] = np.max(dataso[b]) - NormMini
    NormMin[i] = NormMini/NormDlt[i] + 0.5

    datas[b] = dataso[b]/NormDlt[i] - NormMin[i]

## refit space
data_delta = pd.read_csv('./deltas.csv')

# separate breakpoints and output
refitso = data_delta[brkpts].astype(np.float64)
refitf  = data_delta[output].astype(np.float64)

# make this nondimensional
refits = refitso.copy()
for i, b in enumerate(brkpts):
    refits[b] = refitso[b]/NormDlt[i] - NormMin[i]




### Define kernel and model
if method == 'gpr.scikit':

    ## TRAINING
    # Define the kernel parameters - will be overwritten in case of optimisation
    if not if_train_aniso:
        kernel = 1.**2 * RBF(length_scale=0.1) + \
                 0.1**2 * RBF(length_scale=1.) + \
                 0.01**2 * RBF(length_scale=10.)
    else:
        # It must be one length scale per variable!
        kernel = 1.**2 * RBF(length_scale=[1., 1., 1.])

    # Use a kernel info if provided, otherwise optimise
    if if_train_optim:
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=12)
    else:
        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None)

    # Train model
    gpr.fit(datas, dataf)
    mean = gpr.predict(datas, return_cov=False)

    ## TRANSFERING
    # Predict on refit space and compute delta
    mean_at_refits = gpr.predict(refits, return_cov=False)
    delta_means = refitf - mean_at_refits

    # Define the kernel parameters - will be overwritten in optimisation
    kernel = 1.**2 * RBF(length_scale=0.1) + \
             0.1**2 * RBF(length_scale=1.) + \
             0.01**2 * RBF(length_scale=10.)

    gpr_refit = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=12)
    gpr_refit.fit(refits, delta_means)

    # Predict delta on original training space and compute the refitted mean
    refit_delta_means = gpr_refit.predict(datas, return_cov=False)
    refit_mean = dataf + refit_delta_means

    msg = "Training Kernel: " + str(gpr.kernel_)
    print(msg)
    flightlog.write(msg+'\n')

    msg = "Refit Kernel: " + str(gpr_refit.kernel_)
    print(msg)
    flightlog.write(msg+'\n')



elif method == 'gpr.gpflow':

    ## TRAINING
    # Define the kernel parameters - will be overwritten in case of optimisation
    if not if_train_aniso:
        kernel = gpflow.kernels.SquaredExponential(variance=1.**2, lengthscales=0.1) + \
                 gpflow.kernels.SquaredExponential(variance=0.1**2, lengthscales=1.) + \
                 gpflow.kernels.SquaredExponential(variance=0.01**2, lengthscales=10.)
    else:
        kernel = gpflow.kernels.SquaredExponential(variance=1.**2, lengthscales=[1., 1., 1.])

    opt = gpflow.optimizers.Scipy()

    if if_train_optim:
        # Step 1: Make an initial guess with a reduced number of points
        r_datas, r_dataf = reduce_point_cloud(datas.to_numpy(), dataf.to_numpy().reshape(-1,1), target_fraction=r_numberofpoints)
        r_gpr = gpflow.models.GPR(data=(r_datas, r_dataf), kernel=kernel, noise_variance=None)
        r_gpr.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
        gpflow.set_trainable(r_gpr.likelihood.variance, False)

        opt.minimize(r_gpr.training_loss, variables=r_gpr.trainable_variables, options=options)

        msg = "Training Kernel - initial condition: " + str(generate_gpflow_kernel_code(r_gpr.kernel))
        print(msg)
        flightlog.write(msg+'\n')

        # Step 2: Use the optimized parameters as priors for the full model
        optimized_variance = r_gpr.kernel.variance.numpy()
        optimized_lengthscales = r_gpr.kernel.lengthscales.numpy()

        # Set priors
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.math.log(gpflow.utilities.to_default_float(optimized_variance)), 1.0
        )
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(gpflow.utilities.to_default_float(optimized_lengthscales)), 1.0
        )

        # Create the full GPR model
        gpr = gpflow.models.GPR(data=(datas.to_numpy(), dataf.to_numpy().reshape(-1, 1)), kernel=kernel, noise_variance=None)
        gpr.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
        gpflow.set_trainable(gpr.likelihood.variance, False)

        # Optimize the full model
        opt.minimize(gpr.training_loss, variables=gpr.trainable_variables, options=options)

        msg = "Training Kernel: " + str(generate_gpflow_kernel_code(gpr.kernel))
        print(msg)
        flightlog.write(msg+'\n')

    else:
        gpr = gpflow.models.GPR(data=(datas.to_numpy(), dataf.to_numpy().reshape(-1,1)), kernel=kernel, noise_variance=None)
        gpr.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
        gpflow.set_trainable(gpr.likelihood.variance, False)

    # store the posterior for faster prediction
    posterior_gpr = gpr.posterior()

    # Predict on refit space and compute delta
    mean = posterior_gpr.predict_f(datas.to_numpy())[0].numpy().reshape(-1)

    ## TRANSFERING
    # Predict on refit space and compute delta
    mean_at_refits = posterior_gpr.predict_f(refits.to_numpy())[0].numpy().reshape(-1)
    delta_means = refitf - mean_at_refits

    # Define a dummy kernel - the focus is the mean function, so the kernel is frozen and the mean function trained
    # We do so cause we only have two points to retrain
    # We use a dummy kernel so we can use the gpr tools all the same
    kernel = gpflow.kernels.Constant(gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12)))
    mean_function = gpflow.functions.Linear(A=np.zeros((3, 1)), b=np.zeros(1))

    gpr_refit = gpflow.models.GPR(data=(refits.to_numpy(), delta_means.to_numpy().reshape(-1,1)), kernel=kernel, noise_variance=None, mean_function=mean_function)
    gpr_refit.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
    gpflow.set_trainable(gpr_refit.likelihood.variance, False)
    gpflow.set_trainable(gpr_refit.kernel, False)

    # Optimise the mean function coefficients
    opt.minimize(gpr_refit.training_loss, variables=gpr_refit.trainable_variables, options=options)

    # store the posterior for faster prediction
    posterior_gpr_refit = gpr_refit.posterior()

    # Predict delta on original training space and compute the refitted mean
    refit_delta_means = posterior_gpr_refit.predict_f(datas.to_numpy())[0].numpy().reshape(-1)
    refit_mean = dataf + refit_delta_means

    msg = "Refit Kernel: " + str(generate_gpflow_kernel_code(gpr_refit.kernel))
    print(msg)
    flightlog.write(msg+'\n')



elif method == 'gpr.gpytorch':
    print('Sorry. Nothing here.')



elif method == 'nn.tf':

    trained_model_file = 'model_training_' + '.keras'

    ## TRAINING
    if if_train_optim:
        tmodel = keras.Sequential(
            [layers.Dense(3),
                layers.Dense(1024, activation='elu', kernel_initializer='he_normal'),
            layers.Dense(1)]
            )

        tmodel.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001))

        tmodel.fit(
            datas.to_numpy(),
            dataf.to_numpy(),
            verbose=0, epochs=20000, batch_size=64,
            )

        # store the model for reuse
        tmodel.save(trained_model_file)
    
    else:
        # We simply insert the input data into the kernel
        tmodel = tf.keras.models.load_model(trained_model_file)

    # Predict on refit space and compute delta
    mean = tmodel.predict(datas.to_numpy()).reshape(-1)

    ## TRANSFERING
    # Predict on refit space and compute delta
    mean_at_refits = tmodel.predict(refits.to_numpy()).reshape(-1)
    delta_means = refitf - mean_at_refits

    # Define a dummy kernel - the focus is the mean function, so the kernel is frozen and the mean function trained
    # We do so cause we only have two points to retrain
    # We use a dummy kernel so we can use the gpr tools all the same
    kernel = gpflow.kernels.Constant(gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12)))
    mean_function = gpflow.functions.Linear(A=np.zeros((3, 1)), b=np.zeros(1))

    gpr_refit = gpflow.models.GPR(data=(refits.to_numpy(), delta_means.to_numpy().reshape(-1,1)), kernel=kernel, noise_variance=None, mean_function=mean_function)
    gpr_refit.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
    gpflow.set_trainable(gpr_refit.likelihood.variance, False)
    gpflow.set_trainable(gpr_refit.kernel, False)

    # Optimise the mean function coefficients
    opt = gpflow.optimizers.Scipy()
    opt.minimize(gpr_refit.training_loss, variables=gpr_refit.trainable_variables, options=options)

    # store the posterior for faster prediction
    posterior_gpr_refit = gpr_refit.posterior()

    # Predict delta on original training space and compute the refitted mean
    refit_delta_means = posterior_gpr_refit.predict_f(datas.to_numpy())[0].numpy().reshape(-1)
    refit_mean = dataf + refit_delta_means

    msg = "Refit Kernel: " + str(generate_gpflow_kernel_code(gpr_refit.kernel))
    print(msg)
    flightlog.write(msg+'\n')




### plotting

# contours
param3_range = [500, 800]
for v in param3_range:
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Param3 "+str(v), fontsize=14)
    ax = fig.add_subplot(111)

    # filtering the slice
    filtered_indices = dataso[dataso['param3'] == v].index

    # filter the trained mean - we need a pandas dataframe here
    mean_pd = dataf.copy()
    mean_pd.loc[:] = mean

    # prepare the arrays
    X = np.unique(dataso.loc[filtered_indices]['param1'])
    Y = np.unique(dataso.loc[filtered_indices]['param2'])
    Z1 = dataf.loc[filtered_indices].to_numpy().reshape(len(Y), len(X))
    Z2 = mean_pd.loc[filtered_indices].to_numpy().reshape(len(Y), len(X))
    Z3 = refit_mean.loc[filtered_indices].to_numpy().reshape(len(Y), len(X))

    # define the levels and plot
    levels = np.arange(0.8,0.96,0.02)

    COU = plt.contour(X, Y, Z1, levels=levels, linestyles='solid'  , linewidths=1)
    COF = plt.contour(X, Y, Z2, levels=levels, linestyles='dashed' , linewidths=0.5)
    COD = plt.contour(X, Y, Z3, levels=levels, linestyles='dotted' , linewidths=1)

    plt.clabel(COU, fontsize=9)
    plt.clabel(COD, fontsize=9)

    lines = [
        Line2D([0], [0], color='black', linestyle='solid', linewidth=1),
        Line2D([0], [0], color='black', linestyle='dashed', linewidth=0.5),
        Line2D([0], [0], color='black', linestyle='dotted', linewidth=1)
    ]
    labels = ['', ' fitted', 'refit']
    plt.legend(lines, labels)

    ax.set_xlabel('param1 [-]')
    ax.set_ylabel('param2 [Nm/rad]')

    plt.savefig('the_contours_for_'+str(v)+'.png')
    plt.close()


# X-Ys
cases_param1_param2 = [['c1', 5300, 13.9], ['c2', 11120, 74]]
for c in cases_param1_param2:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    c_nam = c[0]
    c_param1 = c[1]
    c_trq = c[2]

    fig.suptitle("Condition  "+str(c_nam), fontsize=14)

    param3_range = np.linspace(500,900,100)

    # create the X dimension to be fitted
    Xo = pd.DataFrame( {col: [pd.NA] * len(param3_range) for col in datas.columns} )
    Xo['param1'] = c_param1
    Xo['param2'] = c_trq
    Xo['param3'] = param3_range

    X = Xo.copy()
    for i, b in enumerate(brkpts):
        X[b] = Xo[b]/NormDlt[i] - NormMin[i]

    if method == 'scikit':
        Y1 = gpr.predict(X, return_cov=False)
        Y2 = Y1 + gpr_refit.predict(X, return_cov=False)
    elif method == 'gpflow':
        Y1 = posterior_gpr.predict_f(X.to_numpy())[0].numpy().reshape(-1)
        Y2 = Y1 + posterior_gpr_refit.predict_f(X.to_numpy())[0].numpy().reshape(-1)

    # get the closest of the function data
    df = pd.DataFrame(dataso)
    df['distance'] = np.sqrt((df['param1'] - c_param1)**2 + (df['param2'] - c_trq)**2)

    # Get the row(s) with the smallest distance
    closest_points_index = df.loc[df['distance'] == df['distance'].min()].index
    XR = dataso.loc[closest_points_index]['param3']
    FR = dataf[closest_points_index]

    # plot
    plt.plot(param3_range, Y1.T, lw=0.5, label=' fitted')
    plt.plot(param3_range, Y2.T, lw=0.5, ls='--', label='refit')
    plt.scatter(XR, FR.T, lw=0.5, marker='o', label=' closest')

    for i, (x, y) in enumerate(zip(XR, FR.T)):
        label = f"param1={dataso.loc[closest_points_index[i], 'param1']}, param2={dataso.loc[closest_points_index[i], 'param2']}"
        plt.text(x, y, label, fontsize=8, ha='right', va='bottom')

    ax.set_xlabel('param3')
    ax.set_ylabel('var1')

    plt.legend()

    plt.savefig('the_plot_for_'+str(c_nam)+'.png')
    plt.close()


msg = f"Elapsed time: {time.time() - start_time:.2f} seconds"
print(msg)
flightlog.write(msg+'\n')

