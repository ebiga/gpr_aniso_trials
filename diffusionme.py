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

    def _get_mesh_indices_from_flattened(self, flat_id):
        ny, nz = self.shape_full[1], self.shape_full[2]

        i_x = tf.math.floordiv(flat_id, ny * nz)
        i_y = tf.math.floordiv(tf.math.floormod(flat_id, ny * nz), nz)
        i_z = tf.math.floormod(flat_id, nz)

        return i_x, i_y, i_z

    def _get_flat_index(self, x, y, z, shape):
        return x * shape[1] * shape[2] + y * shape[2] + z

    def train_step(self, center_indices):
        nx, ny, nz = self.shape_full[0], self.shape_full[1], self.shape_full[2]
        center_indices = tf.convert_to_tensor(center_indices)

        with tf.GradientTape() as tape:
            #_ mean absolute error
            x_batch = tf.gather(self.DATAX, center_indices)
            y_batch = tf.gather(self.DATAF, center_indices)
            pred = self.base_model(x_batch, training=True)
            loss_e = tf.reduce_mean(tf.abs(tf.squeeze(pred) - tf.squeeze(y_batch)))

            #_ diffusion loss
            #__ create small meshes of the neighbouring points of the batched node
            #__ these meshes are required for the diffusion computations
            lap_pred_list = []
            lap_true_list = []

            for _, id in enumerate(center_indices):
                id_x, id_y, id_z = self._get_mesh_indices_from_flattened(id)

                #-- avoid boundary elements of the reference training mesh for this
                if (1 <= id_x < nx - 1) and (1 <= id_y < ny - 1) and (1 <= id_z < nz - 1):

                    #--- gather local staggered points for Laplacian
                    staggered_mesh = []
                    for dx in [-1, 0]:
                        for dy in [-1, 0]:
                            for dz in [-1, 0]:
                                stag_x, stag_y, stag_z = id_x + dx, id_y + dy, id_z + dz
                                stag_flat = self._get_flat_index(stag_x, stag_y, stag_z, (nx - 1, ny - 1, nz - 1))
                                staggered_mesh.append(self.STAGX[stag_flat])

                    #--- gather local neighbouring points for Laplacian
                    neighbour_mesh = []
                    for dx in [-1, 1]:
                        for dy in [-1, 1]:
                            for dz in [-1, 1]:
                                neig_x, neig_y, neig_z = id_x + dx, id_y + dy, id_z + dz
                                neig_flat = self._get_flat_index(neig_x, neig_y, neig_z, (nx - 1, ny - 1, nz - 1))  ###!!!! confirm
                                neighbour_mesh.append(self.DATAX[neig_flat])

                    #--- make predictions of the meshes
                    staggered_mesh = tf.stack(staggered_mesh)
                    neighbour_mesh = tf.stack(neighbour_mesh)

                    pred_stag = self.base_model(staggered_mesh, training=True)
                    pred_neig = self.base_model(neighbour_mesh, training=True)

                    #--- compute the laplacian
                                                                                                        ###!!!! shapes
                    lap_approx = tf_compute_Laplacian(pred_neig, pred_stag, self.select_dimension)
                    lap_pred_list.append(lap_approx)

                    #--- store the reference Laplacians for this batch, exact the the batch nodes
                    lap_target = self.LAPLF[id]
                    lap_true_list.append(lap_target)

            lap_pred_tensor = tf.stack(lap_pred_list)
            lap_true_tensor = tf.stack(lap_true_list)
            loss_m = self.loss_m_weight * tf.reduce_mean(tf.square(lap_true_tensor - lap_pred_tensor))

            loss = loss_e + loss_m

            #_ regularisation
            reg_loss = tf.add_n(self.base_model.losses) if self.base_model.losses else 0.0
            loss += reg_loss

        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "mae": loss_e, "diff": loss_m}

    @property
    def metrics(self):
        return [self.loss_tracker]


def generate_center_indices(shape):
    nx, ny, nz = shape
    indices = []
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            for z in range(1, nz - 1):
                flat_id = x * ny * nz + y * nz + z
                indices.append(flat_id)
    return np.array(indices, dtype=np.int32)


def NN_training_laplacian(model, DATAX, DATAF, STAGX, LAPLF,
                        shape_train_mesh, select_dimension,
                        trained_model_file, histories, casesetup, flightlog):

    # get the user inputs from Jason
    keras_options = casesetup['keras_setup']
    center_indices = generate_center_indices(shape_train_mesh)

    LAPLF = LAPLF.flatten()

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