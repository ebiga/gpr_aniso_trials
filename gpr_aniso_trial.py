import numpy as np
import pandas as pd
import sklearn
import silence_tensorflow.auto
import gpflow
import time
import torch
import gpytorch
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import gpflow.utilities as gputil
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from matplotlib.lines import Line2D
from tensorflow import keras
from keras import layers

gpflow.config.set_default_float('float64')
torch.set_default_dtype(torch.float64)

tf.random.set_seed(42)
keras.utils.set_random_seed(42)
torch.manual_seed(42)


# Optimisation options
gpflow_options = {
    "maxiter": 1500,
    "gtol": 1e-9,
    "ftol": 1e-9,
}

keras_options = {
    "learning_rate": 0.002,
    "epochs": 1250,
    "batch_size": 32,
    "hidden_layers": 1024,
}

gpytorch_options = {
    "maxiter": 1500,
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
        unique_vals = np.unique( np.round(X[:, dim], decimals=6) )
        reduced_count = max(1, int(len(unique_vals) * per_dim_reduction))
        reduced_vals = unique_vals[:: len(unique_vals) // reduced_count][:reduced_count]
        mask &= np.isin(X[:, dim], reduced_vals)

    # Apply mask to reduce the dataset
    X_reduced = X[mask]
    Y_reduced = Y[mask]

    return X_reduced, Y_reduced


# Compute the rms of the mean
def check_mean(mean, refd):
    delta = refd - mean

    Ntotal = np.shape(delta)[0]

    rms_check = np.sqrt( np.sum( delta**2. )/Ntotal )
    mae_check = np.sum( np.abs(delta) )/Ntotal
    max_check = np.max( np.abs(delta) )

    msg = "Training errors: rms, mean, max: " + f"\t{rms_check:.3e};\t {mae_check:.3e};\t {max_check:.3e}\n"
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
                pred = model(torch.from_numpy(X))
            return pred.mean.numpy().reshape(-1)
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")


# GPYTorch loves a class, doesn't it
class GridGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, grid, train_x, train_y, likelihood):
        super(GridGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridKernel(gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=Ndimensions, lengthscale=torch.tensor(np.full(Ndimensions, 1.0))), outputscale=1.0**2), grid=grid)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



flightlog = open('log.txt', 'w')
start_time = time.time()




### USER OPTIONS

# there are three sizes to pick from:
#  full, mid, small
select_input_size = 'small'

# fitting methods available:
#  'gpr.scikit': GPR by scikit-learn
#  'gpr.gpflow': GPR by GPFlow
#  'gpr.gpytorch' (not available)
#  'nn.tf': Neural network by tensorflow/keras
#  'at.tf': Neural network by tensorflow/keras with an attention layer
method = 'at.tf'

# let the hyperparameters by optmised? otherwise, you must provide them
if_train_optim = True

# if available, allow for anisotropy in the GPR
if_train_aniso = True

# ratio of number of points for a reduced-set initial condition
r_numberofpoints = 0.5




### DATA POINTS

## training space
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

Ndimensions = 3 # first 3 columns have the breakpoints

# separate breakpoints and output
brkpts = data_base.columns[:Ndimensions].to_numpy()
output = data_base.columns[Ndimensions]

dataso = data_base[brkpts].astype(np.float64)
dataf  = data_base[output].astype(np.float64)

# make this nondimensional
NormMin = np.full(Ndimensions, 0.)
NormDlt = np.full(Ndimensions, 1.)
datas = dataso.copy()
for i, b in enumerate(brkpts):
    NormMini   = np.min(dataso[b])
    NormDlt[i] = np.max(dataso[b]) - NormMini
    NormMin[i] = NormMini/NormDlt[i] + 0.5

    datas[b] = dataso[b]/NormDlt[i] - NormMin[i]




### TRAIN THE MODELS

if method == 'gpr.scikit':
    loss = []

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
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=12)
    else:
        model = GaussianProcessRegressor(kernel=kernel, optimizer=None)

    # Train model
    model.fit(datas, dataf, optimizer=lambda obj_func, initial_theta, bounds: 
        [loss.append(val[1]) or val for val in [obj_func(initial_theta)]][0])
    
    # Predict and evaluate
    mean = my_predicts(model, datas.to_numpy())
    flightlog.write(check_mean(mean, dataf.to_numpy()))

    msg = "Training Kernel: " + str(model.kernel_)
    print(msg)
    flightlog.write(msg+'\n')



elif method == 'gpr.gpflow':
    loss = []

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

        opt.minimize(r_gpr.training_loss, variables=r_gpr.trainable_variables, options=gpflow_options)

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
        model = gpflow.models.GPR(data=(datas.to_numpy(), dataf.to_numpy().reshape(-1,1)), kernel=kernel, noise_variance=None)
        model.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
        gpflow.set_trainable(model.likelihood.variance, False)

        # Optimize the full model
        opt.minimize(model.training_loss, variables=model.trainable_variables, options=gpflow_options, step_callback=lambda step, var, val: loss.append(val))

        msg = "Training Kernel: " + str(generate_gpflow_kernel_code(model.kernel))
        print(msg)
        flightlog.write(msg+'\n')

    else:
        model = gpflow.models.GPR(data=(datas.to_numpy(), dataf.to_numpy().reshape(-1,1)), kernel=kernel, noise_variance=None)
        model.likelihood.variance = gpflow.Parameter(1e-10, transform=gpflow.utilities.positive(lower=1e-12))
        gpflow.set_trainable(model.likelihood.variance, False)

    # store the posterior for faster prediction
    posterior_gpr = model.posterior()

    # Predict and evaluate
    mean = my_predicts(posterior_gpr, datas.to_numpy())
    flightlog.write(check_mean(mean, dataf.to_numpy()))



elif method == 'gpr.gpytorch':
    loss = []
    trained_model_file = 'model_training_' + '.pth'

    # Build a gpytorch _grid_ to benefit from the grid method
    grid_sizes  = [NgridX, NgridY, NgridZ]
    grid_bounds = [(min(datas.to_numpy()[:,0]), max(datas.to_numpy()[:,0])), (min(datas.to_numpy()[:,1]), max(datas.to_numpy()[:,1])), (min(datas.to_numpy()[:,2]), max(datas.to_numpy()[:,2]))]
    grid = gpytorch.utils.grid.create_grid(grid_sizes, grid_bounds, extend=False, dtype=torch.float64)

    # Convert data to torch tensors
    train_x = gpytorch.utils.grid.create_data_from_grid(grid)
    train_y = torch.tensor(dataf.to_numpy())

    if if_train_optim:

        # Define the model
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.tensor(np.full(len(train_y),1e-6)))
        model = GridGPRegressionModel(grid, train_x, train_y, likelihood)

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

        msg = "Lengthscale: " + str(model.covar_module.base_kernel.base_kernel.lengthscale.squeeze().tolist()) + "\n" \
            + "Variance: " + str(model.covar_module.base_kernel.outputscale.item()) + "\n"
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
    mean = my_predicts(model, datas.to_numpy())
    flightlog.write(check_mean(mean, dataf.to_numpy()))



elif method == 'nn.tf':
    loss = []
    trained_model_file = 'model_training_' + '.keras'

    # Setup the neural network
    if if_train_optim:
        model = keras.Sequential([
            layers.Input(shape=(Ndimensions,)),
            layers.Dense(Ndimensions),
                layers.Dense(keras_options["hidden_layers"], activation='elu', kernel_initializer='he_normal'),
            layers.Dense(1)
            ])

        model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(learning_rate=keras_options["learning_rate"]))

        history = model.fit(
            datas.to_numpy(),
            dataf.to_numpy(),
            verbose=0, epochs=keras_options["epochs"], batch_size=keras_options["batch_size"],
            )
        loss = history.history['loss']

        # store the model for reuse
        model.save(trained_model_file)
    
    else:
        # We simply insert the input data into the kernel
        model = tf.keras.models.load_model(trained_model_file)

    # Predict and evaluate
    mean = my_predicts(model, datas.to_numpy())
    flightlog.write(check_mean(mean, dataf.to_numpy()))



elif method == 'at.tf':
    loss = []
    trained_model_file = 'model_training_att' + '.keras'

    # Setup the neural network
    if if_train_optim:
        inputs = layers.Input(shape=(Ndimensions,))

        # Expand to (batch_size, Ndimensions)
        re_inputs = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)

        # Apply Multi-Head Attention
        attention_output = layers.MultiHeadAttention(num_heads=1, key_dim=Ndimensions)(re_inputs, re_inputs)

        # Squeeze back to (batch_size, Ndimensions)
        attention_output = layers.Lambda(lambda x: tf.squeeze(x, axis=1), output_shape=(None, Ndimensions))(attention_output)

        # Fully connected layers
        dense_output = layers.Dense(keras_options["hidden_layers"], activation="elu", kernel_initializer='he_normal')(attention_output)
        final_output = layers.Dense(1)(dense_output)

        # Create model
        model = keras.models.Model(inputs=inputs, outputs=final_output)

        model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(learning_rate=keras_options["learning_rate"]))

        history = model.fit(
            datas.to_numpy(),
            dataf.to_numpy(),
            verbose=0, epochs=keras_options["epochs"], batch_size=keras_options["batch_size"],
            )
        loss = history.history['loss']

        # store the model for reuse
        model.save(trained_model_file)
    
    else:
        # We simply insert the input data into the kernel
        model = tf.keras.models.load_model(trained_model_file)

    # Predict and evaluate
    mean = my_predicts(model, datas.to_numpy())
    flightlog.write(check_mean(mean, dataf.to_numpy()))




### PLOTTING

# training convergence
plt.plot(np.array(loss), label='Training Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Convergence')
plt.legend()
plt.savefig('convergence_'+str(method)+'.png')
plt.close()


# contours
param3_range = [0.7, 0.8]

for v in param3_range:
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Param3 "+str(round(v,3)), fontsize=14)
    ax = fig.add_subplot(111)

    # filtering the slice
    filtered_indices = dataso[ np.round(dataso['param3'], decimals=6) == v].index

    # filter the trained mean - we need a pandas dataframe here
    mean_pd = dataf.copy()
    mean_pd.loc[:] = mean

    # prepare the arrays
    X = np.unique( np.round(dataso.loc[filtered_indices]['param1'], decimals=6) )
    Y = np.unique( np.round(dataso.loc[filtered_indices]['param2'], decimals=6) )

    Z1 = dataf.loc[filtered_indices].to_numpy().reshape(len(Y), len(X))
    Z2 = mean_pd.loc[filtered_indices].to_numpy().reshape(len(Y), len(X))

    # define the levels and plot
    levels = np.arange(0.04,0.2,0.02)

    COU = plt.contour(X, Y, Z1, levels=levels, linestyles='solid'  , linewidths=1)
    COF = plt.contour(X, Y, Z2, levels=levels, linestyles='dashed' , linewidths=0.5)

    plt.clabel(COU, fontsize=9)

    lines = [
        Line2D([0], [0], color='black', linestyle='solid' , linewidth=1.0),
        Line2D([0], [0], color='black', linestyle='dashed', linewidth=0.5),
    ]
    labels = ['ref', 'fitted']
    plt.legend(lines, labels)

    ax.set_xlabel('param1')
    ax.set_ylabel('param2')

    plt.savefig('the_contours_for_'+str(v)+'.png')
    plt.close()


# X-Ys
cases_param1_param2 = [['c1', 13.25, 1.39], ['c2', 27.8, 7.4]]
param3_range = np.linspace(0.55,1.0,100)

for c in cases_param1_param2:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    c_name = c[0]

    # get the closest of the function data
    df = pd.DataFrame(dataso)
    df['distance'] = np.sqrt((df['param1'] - c[1])**2 + (df['param2'] - c[2])**2)
    closest_points_index = df.loc[df['distance'] == df['distance'].min()].index

    # get the scattered points closest to the references
    XR = dataso.loc[closest_points_index]['param3']
    FR = dataf[closest_points_index]

    # Fit the data to generate the plot
    c_param1 = np.unique( df.loc[df['distance'] == df['distance'].min()]['param1'] ).item()
    c_param2 = np.unique( df.loc[df['distance'] == df['distance'].min()]['param2'] ).item()

    fig.suptitle("Condition  "+str(c_name)+": param1 "+str(round(c_param1,3))+"; param2 "+str(round(c_param2,3)), fontsize=14)

    # create the X dimension to be fitted
    Xo = pd.DataFrame( {col: [pd.NA] * len(param3_range) for col in datas.columns} )
    Xo['param1'] = c_param1
    Xo['param2'] = c_param2
    Xo['param3'] = param3_range

    X = Xo.copy()
    for i, b in enumerate(brkpts):
        X[b] = Xo[b]/NormDlt[i] - NormMin[i]

    Y1 = my_predicts(model, X.to_numpy())

    # plot
    plt.plot(param3_range, Y1.T, lw=0.5, label='fitted')
    plt.scatter(XR, FR.T, lw=0.5, marker='o', label='ref')

    ax.set_xlabel('param3')
    ax.set_ylabel('var1')

    plt.legend()

    plt.savefig('the_plot_for_'+str(c_name)+'.png')
    plt.close()


msg = f"Elapsed time: {time.time() - start_time:.2f} seconds"
print(msg)
flightlog.write(msg+'\n')

