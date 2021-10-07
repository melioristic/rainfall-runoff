#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Created by : Mohit Anand
# Created on : Thu Oct 07 2021 at 12:15:59 PM
# ==========================================================
# __copyright__ = Copyright (c) 2021, Mohit Anand's Project
# __credits__ = [Mohit Anand,]
# __license__ = Private
# __version__ = 0.0.0
# __maintainer__ = Mohit Anand
# __email__ = itsmohitanand@gmail.com
# __status__ = Development
# ==========================================================

import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from RR.cfg import SAVE_FOLDER
from RR.io import read_data

eps = 0.01
learning_rate = 2*1e-3
eta = 0.1

fname = SAVE_FODLER + f"SENN/models/LSTMNoConcept_year_eps_{eps}_eta_{eta}_lr_{learning_rate*10000}xe-4_epoch_{epoch}"

model = tf.keras.models.load_model(fname)


catch_id = 2034

Xd, Yd, elp, flp, slp, area = read_data(catch_id)

Yd = Yd #flow changed to flow per unit area

# Lets do a 80% 10% and 10% split for train and validation and test 
tot_num = Xd.shape[0]


X = np.zeros((Xd.shape[0]-365-1, 365, 3))
Y = np.zeros(Xd.shape[0]-365-1)

for i in range(Xd.shape[0]-365-1):
    X[i, :, 0] = Xd[i:i+365,0]
    X[i, :, 1] = Xd[i:i+365,1]
    X[i, :, 2] = Xd[i:i+365,2]
    
    Y[i] = Yd[365+i]

# Calculate the mean and standard deviation of precipitation, temperature and solar  radiation

mean_X= tf.math.reduce_mean(X, axis = (0,1))
std_X = tf.math.reduce_std(X, axis = (0,1))


batch_size = 2048

tf_data = tf.data.Dataset.from_tensor_slices(
    (X, Y)
).batch(batch_size)


def norm(x, y):
    return (x - mean_X) / std_X, y

def cast(x, y):
    
    return tf.cast(x, dtype= tf.float32), tf.cast(y, dtype= tf.float32)

dataset = tf_data.map(norm).map(cast)


y_true_list = []
y_pred_list = []
parameter_list = []
x_list= []


for x, y in dataset:
    
    parameter, concept, output = model(x)

    parameter_list.append(parameter)
    y_true_list.extend(y)
    y_pred_list.extend(output)
    x_list.append(x)

p = parameter.array(parameter_list)
y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)
x = np.array(x_list)

print(p.shape)
print(y_true.shape)
print(y_pred.shape)
print(x.shape)







# fig, ax = plt.subplots(1,1, figsize = (10,6))
# ax.plot(y_true_list[:365])
# ax.plot(y_pred_list[:365])
# plt.savefig('val_plot_SENN.png')
# plt.close()
    