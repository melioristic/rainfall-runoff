#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Created by : Mohit Anand
# Created on : Tue Oct 05 2021 at 12:37:21 PM
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

class Conv1D(tf.keras.Model):
  
  def __init__(self, **kwargs):
    super(Conv1D, self).__init__(**kwargs)
    
    # Initialise the layers here
    reg_val  = 1.0
    self.l1 = tf.keras.layers.Conv1D(filters=8, kernel_size = 3,  kernel_regularizer=tf.keras.regularizers.L2(reg_val))
    self.l2 = tf.keras.layers.MaxPool1D(2)
    self.l3 = tf.keras.layers.Conv1D(filters=8, kernel_size = 3,  kernel_regularizer=tf.keras.regularizers.L2(reg_val))
    self.l4 = tf.keras.layers.MaxPool1D(2)
    self.l5 = tf.keras.layers.Conv1D(filters=16, kernel_size = 3,  kernel_regularizer=tf.keras.regularizers.L2(reg_val))
    self.l6 = tf.keras.layers.MaxPool1D(2)
    
    ##### new layers 
    self.l7 = tf.keras.layers.Conv1D(filters = 16, kernel_size = 3,  kernel_regularizer=tf.keras.regularizers.L2(reg_val))
    self.l8 = tf.keras.layers.MaxPool1D(2)
    
    self.l9 = tf.keras.layers.Conv1D(filters = 32, kernel_size = 3,  kernel_regularizer=tf.keras.regularizers.L2(reg_val))
    self.l10 = tf.keras.layers.MaxPool1D(2)
    
    #####
    self.l11 = tf.keras.layers.Flatten()
    self.l12 = tf.keras.layers.Dense(1,  kernel_regularizer=tf.keras.regularizers.L2(reg_val))
  
  def call(self, input_tensor):
    
    x = self.l1(input_tensor)
    x = self.l2(x)
    x = self.l3(x)
    x = self.l4(x)
    x = self.l5(x)
    x = self.l6(x)
    x = self.l7(x)
    x = self.l8(x)
    x = self.l9(x)
    x = self.l10(x)
    x = self.l11(x)
    x = self.l12(x)

    return x 
  
  def model(self, input_shape):
    x =  tf.keras.layers.Input(shape = input_shape)
    
    return tf.keras.Model(inputs = [x], outputs = self.call(x))