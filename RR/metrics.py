import tensorflow as tf

class NashSutcliffeEfficiency(tf.keras.metrics.Metric):
    def __init__(self, name="name_of_metric", **kwargs) -> None:
        super(NashSutcliffeEfficiency, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred):
        
        y_true = tf.reshape(y_true, shape = -1)
        y_pred = tf.reshape(y_pred, shape = -1)

        error2 = tf.math.square(y_pred-y_true)
        numerator = tf.math.reduce_sum(error2)

        mean_obs  = tf.math.reduce_mean(y_true)

        diff_mean_obs2 = tf.math.square(y_true-mean_obs)

        denominator = tf.math.reduce_sum(diff_mean_obs2)
   
        self.NSE = 1-(numerator/denominator) 

    def result(self):
        return self.NSE