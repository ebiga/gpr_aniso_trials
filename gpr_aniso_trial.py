import os
import hjson
import silence_tensorflow.auto
import gpflow
import pickle
import time
import torch
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

from matplotlib import cm
from matplotlib.lines import Line2D
from tensorflow import keras
from keras import layers, saving
from gpflow.monitor import Monitor, MonitorTaskGroup

# get my functions
from auxfunctions import *
from optimme      import *
from plot_nn_database import *

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
    if method == 'gpr.gpflow':
        trained_model_file = os.path.join(dafolder, 'model_training_' + method + '.pkl')
    elif method == 'nn.dense':
        trained_model_file = os.path.join(dafolder, 'model_training_' + method + '.keras')

    return trained_model_file


## FUNCTION: Just loads a model file
def load_model_from_file(method, trained_model_file):

    if '.pkl' in trained_model_file:
        with open(trained_model_file, "rb") as f:
            model = pickle.load(f)

    elif '.keras' in trained_model_file:
        model = tf.keras.models.load_model(trained_model_file)

    return model


## FUNCTION: Setup the model to be run and the file to save it
def get_me_a_model(method, DATAX, DATAF):

    #_ GPR: GPFlow
    if method == 'gpr.gpflow':

        #_ variance
        vars, if_train_variance = kernel_variance_whatabouts(casesetup)

        #_ lengthscale
        lens = kernel_lengthscale_whatabouts(casesetup, Ndimensions)

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

        return model

    #_ NN, Dense
    elif method == 'nn.dense':

        nn_layers = casesetup['keras_setup']["hidden_layers"]

        # Setup the neural network
        input_shape = DATAX.shape[1:]

        inputs = tf.keras.Input(shape=input_shape)
        output = build_nn_trunk(inputs, nn_layers)

        # Define the model to get it out in the world
        model = tf.keras.Model(inputs=inputs, outputs=output)

        return model





### USER OPTIONS

start_time = time.time()

with open('./casesetup.hjson', 'r') as casesetupfile:
    casesetup = hjson.load(casesetupfile)

method = casesetup['method']
if_train_optim = casesetup['if_train_optim']
figformat = casesetup['fig_format']

# file locations
dafolder = method
os.makedirs(dafolder, exist_ok=True)

flightlog = open(os.path.join(dafolder, 'log.txt'), 'w')





### DATA POINTS

# Define dimensions, breakpoints and output headers
flap = "0"
mach = "0.2"

brkpts = ["alpha [deg]", "beta [deg]", "ruddervator_LH [deg]"]
params = ["CH_tail_LH_rud"]

Ndimensions = len(brkpts)
Noutputs = len(params)

# Read the whole CSV once
df = pd.read_csv("./_aerodb/ADB_FDM2.1.csv")

# Filter the desired operating condition
df = df[(df["flap [deg]"] == float(flap)) & (df["mach"] == float(mach)) & (df["ruddervator_LH [deg]"] == df["ruddervator_RH [deg]"])]


#!!! Only work with symmetric deflections 
#__ Extract the symmetric cases
df = df[(df["ruddervator_LH [deg]"] == df["ruddervator_RH [deg]"])]

#__ Mirror symmetric cases: right ruddervator will receive the negative of beta
if (df["symmetry"] == 0).any():

    df_mirror = df[df["symmetry"] == 0].copy()
    df_mirror["beta [deg]"] *= -1

    df_mirror["CH_tail_LH_rud"] = df[df["symmetry"] == 0]["CH_tail_RH_rud"].to_numpy()
    df_mirror["CH_tail_RH_rud"] = df[df["symmetry"] == 0]["CH_tail_LH_rud"].to_numpy()

    df = pd.concat([df, df_mirror], ignore_index=True)
    # df = df.drop_duplicates()


# Attribute to the expected arrays
datas = df[brkpts].astype(np.float64)
dataf = df[params].astype(np.float64)





### TRAIN THE MODELS

trained_model_file = model_filename(method, dafolder)


# We need to build a model first
if if_train_optim == 'restart':
    # Read in a saved model from trained_model_file
    loss = None
    model = load_model_from_file(method, trained_model_file)
else:
    model = get_me_a_model(method, datas.to_numpy(), dataf.to_numpy())


# Now we decide what to do with it
if if_train_optim == 'conventional':
    #_ Conventional optimisations
    loss = []
    if 'gpr' in method:
        model, loss = minimise_GPR_LML(method, model, datas.to_numpy(), dataf.to_numpy(),
                                       trained_model_file, loss, casesetup, flightlog)
    elif 'nn' in method:
        model, loss = minimise_NN_RMSE(method, model, datas.to_numpy(), dataf.to_numpy(),
                                       trained_model_file, loss, casesetup, flightlog)

elif if_train_optim == 'nahimgood':
    #_ Just dry run
    print('Nothing here to run dry yet.')
    exit()


# Predict and evaluate
meanf = my_predicts(model, datas.to_numpy())
flightlog.write(check_mean("Training", meanf, dataf.to_numpy()))




### PLOTTING

# training convergence
if if_train_optim == 'conventional':
    plt.plot(np.array(loss), label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log(Loss)')
    plt.title('Loss Convergence')
    plt.legend()
    plt.savefig(os.path.join(dafolder, 'convergence_'+str(method)+'.'+figformat), format=figformat, dpi=1200)
    plt.close()

plot_surfaces(
    model=model,
    datas=datas,
    dataf=dataf,
    my_predicts=my_predicts,
    dafolder=dafolder,
    figformat=figformat,
    deflections=np.arange(-25, 26, 10),
)

plot_ch_vs_deflection(
    model=model,
    datas=datas,
    dataf=dataf,
    my_predicts=my_predicts,
    dafolder=dafolder,
    figformat=figformat,
    alphas=np.arange(-4,17,4),
    beta_fixed=0.0
)

plot_ch_vs_beta(
    model=model,
    datas=datas,
    dataf=dataf,
    my_predicts=my_predicts,
    dafolder=dafolder,
    figformat=figformat,
    deflections=np.arange(-25,26,5),
    alpha_fixed=0.0
)

msg = f"Elapsed time: {time.time() - start_time:.2f} seconds"
print(msg)
flightlog.write(msg+'\n')
