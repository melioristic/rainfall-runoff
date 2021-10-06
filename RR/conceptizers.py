#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Created by : Mohit Anand
# Created on : Wed Oct 06 2021 at 9:37:44 AM
# ==========================================================
# __copyright__ = Copyright (c) 2021, Mohit Anand's Project
# __credits__ = [Mohit Anand,]
# __license__ = Private
# __version__ = 0.0.0
# __maintainer__ = Mohit Anand
# __email__ = itsmohitanand@gmail.com
# __status__ = Development
# ==========================================================

# diffferent conceptizers for SENN

import tensorflow as tf 
from tensorflow.keras import Model 

class NoConceptEncoder(Model):
    def __init__(self, **kwargs):
        super(NoConceptEncoder, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Flatten()

    def call(self, input_tensor):
        
        x = self.l1(input_tensor)
        
        return x
    
    def model(self, input_shape: tuple):
        """
        A model call just to have the input shape visible in summary
        Args:
            input_shape (tuple): Tuple having an input shape
        Returns:
            tf.keras.Model: The model
        """
        x = tf.keras.layers.Input(shape=input_shape)

        return Model(inputs=[x], outputs=self.call(x))