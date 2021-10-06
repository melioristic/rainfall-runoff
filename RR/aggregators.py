import tensorflow as tf
from tensorflow.keras import Model

from RR.parametrizers import *
from RR.conceptizers import *

class LSTMNoConceptAggregator(Model):
    
    def __init__(self,  **kwargs):
        super(LSTMNoConceptAggregator, self).__init__(**kwargs)

        self.conceptizer = NoConceptEncoder()
        self.parametrizer = LSTMParametrizer()
        self.parametrizer.build(input_shape=(None, 365, 3))
        self.parametrizer.call(tf.keras.layers.Input(shape = (365,3)))
        self.parametrizer.summary()


    def call(self, input_tensor):

        concept = self.conceptizer(input_tensor)
        parameter = self.parametrizer(input_tensor)
        
        output = tf.math.reduce_sum(tf.math.multiply(concept, parameter), axis = 1)

        return parameter, concept, output