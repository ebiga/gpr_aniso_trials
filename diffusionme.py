import os
import hjson
import sklearn
import silence_tensorflow.auto
import gpflow
import pickle
import time
import torch
import gpytorch
import scipy

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




## FUNCTION: Compute Laplacians for 2D and 3D, cell centred, for normal or staggered meshes
#_ The Laplacians are computed along the diagonals, so we can have only one staggered mesh
#_ It's never enough any effort to make life easier...
def compute_Laplacian(f_orig, f_stag, select_dimension):

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



## FUNCTION: minimise the diffusion loss
bound = scipy.optimize.Bounds(0.005,500.)

def GPR_training_laplacian(model, DATAX, DATAF, LAPLF, STAGX, select_dimension,
                           shape_train_mesh, shape_stagg_mesh, histories, casesetup, flightlog):

    ## Get the user inputs from Jason
    #_ Optimiser options
    optim_options = casesetup['GPR_setup']['diffusionloss_minimise_setup']

    #_ Kernel variance and lengthscale
    vars, if_train_variance = kernel_variance_whatabouts(casesetup)
    lens = kernel_lengthscale_whatabouts(casesetup, select_dimension)


    ## A function to evaluate the training and diffusion losses within a minimiser
    def evaluate_trial(x, model):

        # Update the kernel with the optimisation variables
        if if_train_variance:
            lens = x[0] if len(x) == 1 else x[:-1]
            vars = x[-1]
        else:
            lens = x[0] if len(x) == 1 else x
            vars = None
        
        update_kernel_params(model, lens, vars)


        # Estimate the loss metric at the staggered mesh
        #_ Training loss
        predf  = my_predicts(model, DATAX)
        loss_e = np.sqrt(np.mean((predf - DATAF)**2.))

        predf_mesh = reshape_flatarray_like_reference_meshgrid(predf, shape_train_mesh, select_dimension)

        #_ Diffusion loss
        predf_staggered     = my_predicts(model, STAGX)
        predf_staggeredmesh = predf_staggered.reshape(shape_stagg_mesh)

        laplacian_predf = compute_Laplacian(predf_mesh, predf_staggeredmesh, select_dimension)

        loss_m = np.sqrt(np.mean((laplacian_predf - LAPLF)**2.))

        loss = loss_e + loss_m
        histories.append([np.log10(loss), np.log10(loss_e), np.log10(loss_m)])

        return loss


    # Optimizesss
    if if_train_variance:
        x0 = [lens, vars] if np.isscalar(lens) else list(lens) + [vars]
    else:
        x0 = [lens] if np.isscalar(lens) else list(lens)

    res = scipy.optimize.minimize(evaluate_trial, x0, args=(model,), method='COBYQA', bounds=bound, options=optim_options)

    msg = "Training Kernel: " + generate_kernel_info(model)
    print(msg)
    flightlog.write(msg+'\n')

    return model




## FUNCTION: minimises the RMSE for NNs
def NN_training_laplacian(model, DATAX, DATAF, STAGX, refLAPLF,
                          shape_train_mesh, shape_stagg_mesh, select_dimension,
                          trained_model_file, loss, casesetup):

    # get the user inputs from Jason
    keras_options = casesetup['keras_setup']

    loss_fn = NN_laplacian_loss(model, STAGX, refLAPLF, shape_train_mesh, shape_stagg_mesh, select_dimension)

    model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(learning_rate=keras_options["learning_rate"]))

    history = model.fit(
        x=DATAX,
        y=DATAF,
        verbose=0, epochs=keras_options["epochs"], batch_size=keras_options["batch_size"],
        )
    loss = np.log(history.history['loss'])

    # store the model for reuse
    model.save(trained_model_file)




## FUNCTION: the proper Laplacian loss
def NN_laplacian_loss(model, STAGX, LAPLF, shape_train_mesh, shape_stagg_mesh, select_dimension):
    def loss_fn(y_true, y_pred):
        loss_e = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

        pred_f_mesh = tf_reshape_flatarray_like_reference_meshgrid(y_pred, shape_train_mesh, select_dimension)

        pred_staggered = model(STAGX, training=False)
        pred_staggered_mesh = tf.reshape(pred_staggered, shape_stagg_mesh)

        lap_pred = compute_Laplacian(pred_f_mesh, pred_staggered_mesh, select_dimension)
        loss_m = tf.sqrt(tf.reduce_mean(tf.square(lap_pred - LAPLF)))

        return loss_e + loss_m
    return loss_fn