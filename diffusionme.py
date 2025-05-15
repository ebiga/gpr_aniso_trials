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

# Tensorflow version - we only use for the staggered computations in the loop
def tf_compute_Laplacian(f_orig, f_stag, select_dimension: str, staggered=True):

    def _laplacian_2d(f_orig, f_stag, staggered=True):
        f_orig = tf.convert_to_tensor(f_orig, dtype=tf.float64)
        f_stag = tf.convert_to_tensor(f_stag, dtype=tf.float64)
        delta = tf.constant(2.0 * 0.5**2 if staggered else 2.0, dtype=tf.float64)

        if staggered:
            d1 = tf.abs(f_orig[2:, 2:] + f_orig[:-2, :-2] - f_stag[1:, 1:] - f_stag[:-1, :-1])
            d1 /= (f_orig[2:, 2:] + f_orig[:-2, :-2] + f_stag[1:, 1:] + f_stag[:-1, :-1]) * (3. * delta)

            d2 = tf.abs(f_orig[:-2, 2:] + f_orig[2:, :-2] - f_stag[:-1, 1:] - f_stag[1:, :-1])
            d2 /= (f_orig[:-2, 2:] + f_orig[2:, :-2] + f_stag[:-1, 1:] + f_stag[1:, :-1]) * (3. * delta)
        else:
            center = f_orig[1:-1, 1:-1]
            d1 = tf.abs(f_orig[2:, 2:] + f_orig[:-2, :-2] - 2. * center)
            d1 /= (f_orig[2:, 2:] + f_orig[:-2, :-2] + 2. * center) * delta

            d2 = tf.abs(f_orig[:-2, 2:] + f_orig[2:, :-2] - 2. * center)
            d2 /= (f_orig[:-2, 2:] + f_orig[2:, :-2] + 2. * center) * delta

        return d1 + d2


    def _laplacian_3d(f_orig, f_stag, staggered=True):
        f_orig = tf.convert_to_tensor(f_orig, dtype=tf.float64)
        f_stag = tf.convert_to_tensor(f_stag, dtype=tf.float64)
        delta = tf.constant(2.0 * 0.5**2 if staggered else 2.0, dtype=tf.float64)

        if staggered:
            d1 = tf.abs(f_orig[2:, 2:, 2:] + f_orig[:-2, :-2, :-2] - f_stag[1:, 1:, 1:] - f_stag[:-1, :-1, :-1])
            d1 /= (f_orig[2:, 2:, 2:] + f_orig[:-2, :-2, :-2] + f_stag[1:, 1:, 1:] + f_stag[:-1, :-1, :-1]) * (3. * delta)

            d2 = tf.abs(f_orig[:-2, 2:, 2:] + f_orig[2:, :-2, :-2] - f_stag[:-1, 1:, 1:] - f_stag[1:, :-1, :-1])
            d2 /= (f_orig[:-2, 2:, 2:] + f_orig[2:, :-2, :-2] + f_stag[:-1, 1:, 1:] + f_stag[1:, :-1, :-1]) * (3. * delta)

            d3 = tf.abs(f_orig[2:, :-2, 2:] + f_orig[:-2, 2:, :-2] - f_stag[1:, :-1, 1:] - f_stag[:-1, 1:, :-1])
            d3 /= (f_orig[2:, :-2, 2:] + f_orig[:-2, 2:, :-2] + f_stag[1:, :-1, 1:] + f_stag[:-1, 1:, :-1]) * (3. * delta)
        else:
            center = f_orig[1:-1, 1:-1, 1:-1]
            d1 = tf.abs(f_orig[2:, 2:, 2:] + f_orig[:-2, :-2, :-2] - 2. * center)
            d1 /= (f_orig[2:, 2:, 2:] + f_orig[:-2, :-2, :-2] + 2. * center) * delta

            d2 = tf.abs(f_orig[:-2, 2:, 2:] + f_orig[2:, :-2, :-2] - 2. * center)
            d2 /= (f_orig[:-2, 2:, 2:] + f_orig[2:, :-2, :-2] + 2. * center) * delta

            d3 = tf.abs(f_orig[2:, :-2, 2:] + f_orig[:-2, 2:, :-2] - 2. * center)
            d3 /= (f_orig[2:, :-2, 2:] + f_orig[:-2, 2:, :-2] + 2. * center) * delta

        return d1 + d2 + d3

    if select_dimension == "3D":
        return _laplacian_3d(f_orig, f_stag, staggered)
    else:
        return _laplacian_2d(f_orig, f_stag, staggered)




## FUNCTION: minimise the diffusion loss
bound = scipy.optimize.Bounds(0.005,500.)

def GPR_training_laplacian(model, DATAX, DATAF, STAGX, LAPLF,
                        shape_train_mesh, shape_stagg_mesh, select_dimension,
                        trained_model_file, histories, casesetup, flightlog):

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

    return model, histories




## CLASS: define a Laplace diffusion loss for a NN model
class LaplacianModel(keras.Model):
    def __init__(self, base_model, DATAX, STAGX, LAPLF, shape_train_mesh, shape_stagg_mesh, select_dimension):
        super().__init__()
        self.base_model = base_model

        self.DATAX = DATAX
        self.STAGX = STAGX
        self.LAPLF = tf.convert_to_tensor(LAPLF)

        self.shape_train_mesh = shape_train_mesh
        self.shape_stagg_mesh = shape_stagg_mesh
        self.select_dimension = select_dimension

        self.loss_tracker = keras.metrics.Mean(name="loss")

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x_batch, y_batch = data

            #_ Training loss
            y_pred_batch = self.base_model(x_batch, training=True)
            loss_e = tf.reduce_mean(tf.square(y_batch - tf.squeeze(y_pred_batch)))

            #_ Diffusion loss
            #__ this is performed in the whole mesh cause the differentiation
            predf = self.base_model(self.DATAX, training=True)
            predf_staggered = self.base_model(self.STAGX, training=True)

            predf_mesh = tf_reshape_flatarray_like_reference_meshgrid(predf, self.shape_train_mesh, self.select_dimension)
            predf_staggeredmesh = tf.reshape(predf_staggered, self.shape_stagg_mesh)

            laplacian_pred = tf_compute_Laplacian(predf_mesh, predf_staggeredmesh, self.select_dimension)

            loss_m = tf.reduce_mean(tf.square(self.LAPLF - laplacian_pred))

            loss = loss_e + loss_m

        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]




## FUNCTION: minimises the RMSE for NNs
def NN_training_laplacian(model, DATAX, DATAF, STAGX, LAPLF,
                        shape_train_mesh, shape_stagg_mesh, select_dimension,
                        trained_model_file, histories, casesetup, flightlog):

    # get the user inputs from Jason
    keras_options = casesetup['keras_setup']

    # give the base model to the Laplacian model
    model = LaplacianModel(
        base_model=model,
        DATAX=DATAX,
        STAGX=STAGX,
        LAPLF=LAPLF,
        shape_train_mesh=shape_train_mesh,
        shape_stagg_mesh=shape_stagg_mesh,
        select_dimension=select_dimension,
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=keras_options["learning_rate"]))

    history = model.fit(
        DATAX,
        DATAF,
        verbose=0, epochs=keras_options["epochs"], batch_size=keras_options["batch_size"],
        )
    histories = np.log(history.history['loss'])

    # store the model for reuse
    model.save(trained_model_file)

    return model, histories
