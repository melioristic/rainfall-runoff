#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Created by : Mohit Anand
# Created on : Wed Oct 06 2021 at 9:45:56 AM
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

def compute_loss_theta(tape, parameter, concept, output, x):

    b = x.shape[0]
    in_dim = (x.shape[1], x.shape[2])

    feature_dim = in_dim[0]*in_dim[1]

    # J = tape.batch_jacobian(concept, x)
    grad_fx = tape.gradient(output, x)
    grad_fx = tf.reshape(grad_fx,shape=(b, feature_dim))
    # J = tf.reshape(J, shape=(b, feature_dim, feature_dim))

    parameter = tf.expand_dims(parameter, axis =1)

    loss_theta_matrix = grad_fx # - tf.matmul(parameter, J)

    loss_theta = tf.norm(loss_theta_matrix)

    return loss_theta