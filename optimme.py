import os
import hjson
import sklearn
import silence_tensorflow.auto
import gpflow
import pickle
import time
import torch
import gpytorch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpflow.utilities as gputil
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from tensorflow import keras
from keras import layers, saving
from gpflow.monitor import Monitor, MonitorTaskGroup
from keras.callbacks import ReduceLROnPlateau

# get my functions
from auxfunctions import *
from auxgpytorch  import *

# set floats and randoms
gpflow.config.set_default_float('float64')
tf.keras.backend.set_floatx('float64')
torch.set_default_dtype(torch.float64)

tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
torch.manual_seed(42)



## FUNCTION: minimises the log-marginal likelihood for GPRs
def minimise_GPR_LML(method, model, DATAX, DATAF, trained_model_file, loss, casesetup, flightlog):

    # get the user inputs from Jason
    gpflow_options = casesetup['GPR_setup']['gpflow_setup']
    gpytorch_options = casesetup['GPR_setup']['gpytorch_setup']


    if method == 'gpr.scikit':
        # Reset the model with the optmization options
        n_restarts_optimizer = casesetup['GPR_setup']['scikit_setup']['n_restarts_optimizer']
        model = GaussianProcessRegressor(kernel=model.kernel, n_restarts_optimizer=n_restarts_optimizer)

        # Train model with proper optimisation
        model.fit(DATAX, DATAF)

        msg = "Training Kernel: " + generate_kernel_info(model)
        print(msg)
        flightlog.write(msg+'\n')

        # store the model for reuse
        with open(trained_model_file, "wb") as f:
            pickle.dump(model, f)

        return model, loss



    elif method == 'gpr.gpflow':
        # Define the optimizer
        opt = gpflow.optimizers.Scipy()

        # Optimize the full model
        monitor = Monitor(MonitorTaskGroup( [lambda x: loss.append(float(model.training_loss()))] ))
        opt.minimize(model.training_loss, variables=model.trainable_variables, options=gpflow_options, step_callback=monitor, compile=True)

        msg = "Training Kernel: " + generate_kernel_info(model)
        print(msg)
        flightlog.write(msg+'\n')

        # store the model for reuse
        with open(trained_model_file, "wb") as f:
            pickle.dump(model, f)

        # store the posterior for faster prediction
        model = model.posterior()

        return model, loss



    elif method == 'gpr.gpytorch':

        train_x = torch.tensor(DATAX)
        train_y = torch.tensor(DATAF)

        # set the mode to training
        model.train()
        model.likelihood.train()

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        # # Use Adam, hey Adam, me again, an apple
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        # for i in range(gpytorch_options['maxiter']):
        #     optimizer.zero_grad()
        #     output = model(train_x)
        #     opt_loss = -mll(output, train_y)
        #     opt_loss.backward()
        #     loss.append(opt_loss.item())
        #     optimizer.step()
        
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=50)
        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            return loss
        for i in range(gpytorch_options['maxiter']):
            loss = optimizer.step(closure)

        msg = "Training Kernel: " + generate_kernel_info(model)
        print(msg)
        flightlog.write(msg)

        # store the model for reuse
        torch.save(model, trained_model_file)

        # set the mode to eval
        model.eval()
        model.likelihood.eval()

        return model, loss



## FUNCTION: minimises the RMSE for NNs
def minimise_NN_RMSE(method, model, DATAX, DATAF, trained_model_file, loss, casesetup, flightlog):

    # get the user inputs from Jason
    keras_options = casesetup['keras_setup']

    # adaptive learning rate for good measure
    callbacks_list = []
    if keras_options['if_learning_rate_schedule']:
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=250, cooldown=50, verbose=1, min_lr=1e-5)
        callbacks_list.append(reduce_lr)

    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(learning_rate=keras_options["learning_rate"]))

    # let's try this babe
    history = model.fit(
        DATAX,
        DATAF,
        verbose=0, epochs=keras_options["epochs"], batch_size=keras_options["batch_size"],
        callbacks=callbacks_list,
        )
    loss = np.log(history.history['loss'])

    # store the model for reuse
    model.save(trained_model_file)

    return model, loss
