#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Created by : Mohit Anand
# Created on : Wed Oct 06 2021 at 9:38:40 AM
# ==========================================================
# __copyright__ = Copyright (c) 2021, Mohit Anand's Project
# __credits__ = [Mohit Anand,]
# __license__ = Private
# __version__ = 0.0.0
# __maintainer__ = Mohit Anand
# __email__ = itsmohitanand@gmail.com
# __status__ = Development
# ==========================================================

# Different parametrizers for SENN

import tensorflow as tf
from tensorflow.keras import Model

class LSTMParametrizer(Model):
    def __init__(self,  **kwargs):
        super(LSTMParametrizer, self).__init__(**kwargs)

        reg_val = tf.keras.regularizers.L2(1.0)

        self.l1 = tf.keras.layers.LSTM(128, kernel_regularizer = reg_val)
        self.l2 = tf.keras.layers.Dense(1095, kernel_regularizer = reg_val)


    def call(self, input_tensor):
        
        x = self.l1(input_tensor)
        x = self.l2(x)
        
        return x 