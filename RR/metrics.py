import tensorflow as tf

class NashSutcliffeEfficiency(tf.keras.metrics.Metric):
    def __init__(self, name="name_of_metric", **kwargs) -> None:
        super(NashSutcliffeEfficiency, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred):
        
        numerator = tf.math.reduce_mean(tf.math.square(y_pred-y_true))
        denominator = tf.math.reduce_mean(tf.math.square(y_true-tf.math.reduce_mean(y_pred)))
   
        self.NSE = 1-numerator/denominator 

    def result(self):
        return self.NSE