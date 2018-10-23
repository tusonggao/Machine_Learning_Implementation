import numpy as np
import tensorflow as tf
import scipy.stats as ss
from sklearn import metrics

class Losses:
    @staticmethod
    def mse(y, pred, _, weights=None):
        if weights is None:
            return tf.losses.mean_squared_error(y, pred)
        return tf.losses.mean_squared_error(y, pred,
                                            tf.reshape(weights, [-1, 1]))

    @staticmethod
    def cross_entropy(y, pred, already_prob, weights=None):
        if already_prob:
            eps = 1e-2
            pred = tf.log(tf.clip_by_value(pred, eps, 1-eps))
        if weights is None:
            return tf.losses.softmax_cross_entropy(y, pred)
        return tf.losses.softmax_cross_entropy(y, pred, weights)

    @staticmethod
    def correlation(y, pred, _, weights=None):
        y_mean, y_var = tf.nn.moments(y, 0)
        pred_mean, pred_var = tf.nn.moments(pred, 0)
        if weights is None:
            e = tf.reduce_mean((y-y_mean)*(pred-pred_mean))
        else:
            e = tf.reduce_mean((y-y_mean)*(pred-pred_mean)*weights)

        return -e/tf.sqrt(y_var*pred_var)