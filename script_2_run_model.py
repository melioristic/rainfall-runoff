#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Created by : Mohit Anand
# Created on : Sun Oct 03 2021 at 3:50:20 PM
# ==========================================================
# __copyright__ = Copyright (c) 2021, Mohit Anand's Project
# __credits__ = [Mohit Anand,]
# __license__ = Private
# __version__ = 0.0.0
# __maintainer__ = Mohit Anand
# __email__ = itsmohitanand@gmail.com
# __status__ = Development
# ==========================================================

from RR.io import read_data
from RR.models import Conv1D

import tensorflow as tf
import numpy as np
import time


from RR.metrics import NashSutcliffeEfficiency
from RR.cfg import SAVE_FOLDER

Xd, Yd, elp, flp, slp, area = read_data()

# Lets do a 80% 10% and 10% split for train and validation and test 
tot_num = Xd.shape[0]


X = np.zeros((Xd.shape[0]-365-1, 365, 3))
Y = np.zeros(Xd.shape[0]-365-1)

Y = Y/area #flow changed to flow per unit area



for i in range(Xd.shape[0]-365-1):
    X[i, :, 0] = Xd[i:i+365,0]
    X[i, :, 1] = Xd[i:i+365,1]
    X[i, :, 2] = Xd[i:i+365,2]
    
    Y[i] = Yd[365+i]

# Calculate the mean and standard deviation of precipitation, temperature and solar  radiation

mean_X= tf.math.reduce_mean(X, axis = (0,1))
std_X = tf.math.reduce_std(X, axis = (0,1))

end_train = int(0.80*tot_num)
strt_val = int(0.80*tot_num)
end_val = int(0.90*tot_num)
strt_test = int(0.90*tot_num)

X_train = X[:end_train,:,:]
X_val = X[strt_val:end_val,:,:]
X_test = X[strt_test:,:,:]

Y_train = Y[:end_train]
Y_val = Y[strt_val:end_val]
Y_test = Y[strt_test:]


batch_size = 128


x_spec = tf.TensorSpec(shape = (None, 365, 3), dtype = tf.float32)
y_spec = tf.TensorSpec(shape = (None, 1), dtype = tf.float32)

output_signature= (x_spec, y_spec)

tf_train_data = tf.data.Dataset.from_tensor_slices(
    (X_train, Y_train)
).batch(batch_size)


tf_val_data = tf.data.Dataset.from_tensor_slices(
    (X_val, Y_val)
).batch(batch_size)


def norm(x, y):
    return (x - mean_X) / std_X, y

def cast(x, y):
    
    return tf.cast(x, dtype= tf.float32), tf.cast(y, dtype= tf.float32)

train_dataset = tf_train_data.map(norm).map(cast)

val_dataset = tf_val_data.map(norm).map(cast)

loss_object = tf.keras.losses.MeanSquaredError()
learning_rate = 2 * 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_acc_metric = NashSutcliffeEfficiency()

val_acc_metric = NashSutcliffeEfficiency()

model = Conv1D()

model.model(input_shape=(365, 3)).summary() 

def apply_gradient(optimizer, model, x, y):

    with tf.GradientTape() as tape:
        y_pred = model(x)

        loss_value = loss_object(y_true=y, y_pred=y_pred)

    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return y_pred, loss_value


def train_data_for_one_epoch():

    losses = []
    # pbar = tqdm(total=len(list(enumerate(train_dataset))), position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')
    for step, (x_train, y_train) in enumerate(train_dataset):
        
        y_pred, loss_value = apply_gradient(optimizer, model, x_train, y_train)

        losses.append(loss_value)
        
        train_acc_metric(y_train, y_pred)

    #   pbar.set_description("Training loss for step %s: %.4f" % (int(step), float(loss_value)))
    #    pbar.update()
    return losses

def perform_validation():
    losses = []
    for x_val, y_val in val_dataset:
      
        y_pred = model(x_val)
        val_loss = loss_object(y_true=y_val, y_pred=y_pred)
        losses.append(val_loss)
        val_acc_metric(y_val, y_pred)
    return losses


epochs_val_losses, epochs_train_losses = [], []

strt_time = time.time()

for epoch in range(101):
    # Run through  training batch
    print("Start of epoch %d" % (epoch))

    losses_train = train_data_for_one_epoch()

    train_acc = train_acc_metric.result()

    losses_val = perform_validation()
    val_acc = val_acc_metric.result()

    losses_train_mean = np.mean(losses_train)
    losses_val_mean = np.mean(losses_val)
    epochs_val_losses.append(losses_val_mean)
    epochs_train_losses.append(losses_train_mean)

    fname = f"conv1D/models/conv1D_year_lr_{learning_rate*10000}xe-4_epoch_{epoch}"
    if epoch % 10 == 0:
        fpath = SAVE_FOLDER + fname
        model.save(fpath)

    print(
        "\n Epoch %s: Train loss: %.4f  Validation Loss: %.4f, Train PCC: %.4f, Validation PCC %.4f"
        % (
            epoch,
            float(losses_train_mean),
            float(losses_val_mean),
            float(train_acc),
            float(val_acc),
        )
    )

    train_acc_metric.reset_states()
    val_acc_metric.reset_states()

    print(f"Time taken for epoch {epoch} is {((time.time()-strt_time)/60):.2f} minutes")