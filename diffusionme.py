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
def tf_compute_Laplacian(f_orig, f_stag, select_dimension: str):

    #_ We compute the Laplacian with a 5-point stencil
    grid_spacing = tf.constant(0.5, dtype=tf.float64)

    if select_dimension == "3D":
        delta = 3. * tf.square(grid_spacing)

        # Each pair is a space diagonal of the 2x2x2 cube
        diagonals = [
            ((0, 0, 0), (1, 1, 1)),
            ((0, 1, 0), (1, 0, 1)),
            ((0, 0, 1), (1, 1, 0)),
            ((0, 1, 1), (1, 0, 0)),
        ]

        total = 0.0
        for (a, b) in diagonals:
            o1, o2 = f_orig[a[0], a[1], a[2]], f_orig[b[0], b[1], b[2]]
            s1, s2 = f_stag[a[0], a[1], a[2]], f_stag[b[0], b[1], b[2]]

            total += tf.abs(o1 + o2 - s1 - s2) / (o1 + o2 + s1 + s2)

        return total / (3. * delta)

    else:
        delta = 2. * tf.square(grid_spacing)

        d1 = tf.abs(f_orig[0, 0] + f_orig[1, 1] - f_stag[0, 0] - f_stag[1, 1]) \
                 / (f_orig[0, 0] + f_orig[1, 1] + f_stag[0, 0] + f_stag[1, 1])

        d2 = tf.abs(f_orig[0, 1] + f_orig[1, 0] - f_stag[0, 1] - f_stag[1, 0]) \
                 / (f_orig[0, 1] + f_orig[1, 0] + f_stag[0, 1] + f_stag[1, 0])

        return (d1 + d2) / (3. * delta)




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
    def __init__(self, base_model, DATAX, DATAF, STAGX, LAPLF, shape_full, select_dimension):
        super().__init__()
        self.base_model = base_model
        self.DATAX = tf.convert_to_tensor(DATAX)
        self.DATAF = tf.convert_to_tensor(DATAF)
        self.STAGX = tf.convert_to_tensor(STAGX)
        self.LAPLF = tf.convert_to_tensor(LAPLF)

        self.shape_full = shape_full
        self.select_dimension = select_dimension

        self.loss_tracker = keras.metrics.Mean(name="loss")

        self.loss_m_weight = tf.Variable(0.0, trainable=False, dtype=tf.float64)

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    # Funcions necessary to relate local meshes in flattened and mesh shapes
    def _get_mesh_indices_from_flattened(self, flat_id):
        nx, ny, nz = self.shape_full

        i_z = tf.math.floordiv(flat_id, nx * ny)
        rem = tf.math.floormod(flat_id, nx * ny)
        i_y = tf.math.floordiv(rem, nx)
        i_x = tf.math.floormod(rem, nx)

        return i_x, i_y, i_z
    def _get_flat_index_reversed(self, x, y, z, shape): # for DATAX and csv shiz
        nx, ny, nz = shape
        return z * nx * ny + y * nx + x
    def _get_flat_index(self, x, y, z, shape): # for normal crap
        nx, ny, nz = shape
        return x * ny * nz + y * nz + z

    # This is our wrapper for a tf-based laplacian on local meshes around mini-batch points
    # Each central node will have 8 neighbours from the 3 diagonals (in 3-D). We only take these points and not
    # the whole cube to be as fast as possible, which is quite cool in this approach.
    def compute_local_laplacian(self, id_n):
        nx, ny, nz = self.shape_full
        id_x, id_y, id_z = self._get_mesh_indices_from_flattened(id_n)

        #-- avoid boundary elements of the reference training mesh for this
        cond = tf.reduce_all([
            id_x >= 1, id_x < nx - 1,
            id_y >= 1, id_y < ny - 1,
            id_z >= 1, id_z < nz - 1
        ])

        def inner():
            #--- gather the neighbouring training nodes
            local_neig = [self.DATAX[self._get_flat_index_reversed(id_x + dx, id_y + dy, id_z + dz, (nx, ny, nz))]
                for dx in [-1, 1] for dy in [-1, 1] for dz in [-1, 1]]
            predf = self.base_model(tf.stack(local_neig), training=True)
            predf_mesh = tf.reshape(predf, [2, 2, 2])

            #--- gather local staggered points
            local_stag = [self.STAGX[self._get_flat_index(id_x + dx, id_y + dy, id_z + dz, (nx - 1, ny - 1, nz - 1))]
                for dx in [-1, 0] for dy in [-1, 0] for dz in [-1, 0]]
            predf_staggered = self.base_model(tf.stack(local_stag), training=True)
            predf_staggeredmesh = tf.reshape(predf_staggered, [2, 2, 2])

            #--- over to the Laplacians
            return self.LAPLF[id_x - 1, id_y - 1, id_z - 1] - tf_compute_Laplacian(predf_mesh, predf_staggeredmesh, self.select_dimension), tf.constant(True)

        def zeros():
            return tf.constant(0.0, dtype=tf.float64), tf.constant(False)

        return tf.cond(cond, inner, zeros)

    # Our special train_step implementation that calls the diffusion operator
    def train_step(self, center_indices):
        center_indices = tf.convert_to_tensor(center_indices, dtype=tf.int32)

        with tf.GradientTape() as tape:
            #_ mean absolute error
            x_batch = tf.gather(self.DATAX, center_indices)
            y_batch = tf.gather(self.DATAF, center_indices)
            pred = self.base_model(x_batch, training=True)
            loss_e = tf.reduce_mean(tf.abs(tf.squeeze(pred) - tf.squeeze(y_batch)))

            #_ diffusion loss
            delta_laplacian_tensor, boundary_flag = tf.map_fn(self.compute_local_laplacian, center_indices, dtype=(tf.float64, tf.bool))

            masked_loss = tf.boolean_mask(tf.square(delta_laplacian_tensor), boundary_flag == True)
            loss_m = self.loss_m_weight * tf.reduce_mean(masked_loss)

            #_ assemble total loss
            loss = loss_e + loss_m

            #_ regularisation
            reg_loss = tf.add_n(self.base_model.losses) if self.base_model.losses else 0.0
            loss += reg_loss

        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))

        self.loss_tracker.update_state(loss)
        for metric in self.metrics:
            metric.update_state(y_batch, pred)

        return {"loss": self.loss_tracker.result(), **{m.name: m.result() for m in self.metrics}}

    @property
    def metrics(self):
        return [self.loss_tracker]


def NN_training_laplacian(model, DATAX, DATAF, STAGX, LAPLF,
                        shape_train_mesh, select_dimension,
                        trained_model_file, histories, casesetup, flightlog):

    # get the user inputs from Jason
    keras_options = casesetup['keras_setup']

    # In this setup, instead of passing the mesh array, we pass the indices and let the minibatching
    # work on it. We retrive the mesh from the index inside train_step.
    # With that we can apply minibatching to the Laplacian as well.
    # We must take care with shaping-flattening so the diffusion operator will work.
    center_indices = np.arange(np.prod(shape_train_mesh), dtype=np.int32)

    # give the base model to the Laplacian model
    model = LaplacianModel(
        base_model=model,
        DATAX=DATAX,
        DATAF=DATAF,
        STAGX=STAGX,
        LAPLF=LAPLF,
        shape_full=shape_train_mesh,
        select_dimension=select_dimension,
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=keras_options["learning_rate"]))

    # adaptive learning rates for good measure, incl a ramp for the diffusion loss cause it's just too much
    callbacks_list = []

    if keras_options['if_learning_rate_schedule']:
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=250, cooldown=50, verbose=1, min_lr=1e-5)
        callbacks_list.append(reduce_lr)

    laplstep_ = FixedStepLossMWeight(model, step_every=500, step_size=0.1, max_weight=1.0)
    callbacks_list.append(laplstep_)

    # let's try this babe
    history = model.fit(
        center_indices,
        verbose=0, epochs=keras_options["epochs"], batch_size=keras_options["batch_size"],
        callbacks=callbacks_list,
        )
    histories = np.log(history.history['loss'])

    # store the model for reuse
    model.save(trained_model_file)

    return model, histories




## FUNCTION: Increases loss_m_weight by step_size every `step_every` epochs.
class FixedStepLossMWeight(tf.keras.callbacks.Callback):
    def __init__(self, model_ref, step_every=5, step_size=0.1, max_weight=1.0):
        super().__init__()
        self.model_ref = model_ref
        self.step_every = step_every
        self.step_size  = step_size
        self.max_weight = max_weight
        self.prev_weight = -1.0

    def on_epoch_begin(self, epoch, logs=None):
        step_count = epoch // self.step_every
        new_weight = min(step_count * self.step_size, self.max_weight)
        if new_weight != self.prev_weight:
            print(f"Epoch {epoch+1}: loss_m_weight = {new_weight:.2e}")
            self.prev_weight = new_weight
        self.model_ref.loss_m_weight.assign(new_weight)