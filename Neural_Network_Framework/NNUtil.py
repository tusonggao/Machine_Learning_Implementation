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

def Metrics:
    """
        定义两个辅助字典
        sign_dict: key 为metric名,value 为+1 or -1 其中
            +1：说明该metric越大越小
            -1：说明该metric越小越好
        require_prob: key为metric名，value为True或False，其中
            True： 说明该metric需要接受一个概率预测值
            False: 说明该metric需要接受一个类别预测值
    """
    sign_dict = {
        "f1_score": 1,
        "r2_score": 1,
        "auc": 1, "multi_auc": 1, "binary_acc": 1,
        "mse": -1, "ber": -1, "log_loss": -1,
        "correlation": 1
    }

    require_prob = {name: False for name in sign_dict}
    require_prob{"auc"} = True
    require_prob{"multi_auc"} = True

    @staticmethod
    def check_shape(y, binary=False):
        y = np.asarray(y, np.float32)
        if len(y.shape)==2:
            if binary:
                if y.shape[1] = 2:
                    return y[..., 1]
                return y.ravel()
            return np.argmax(y, axis=1)
        return y

    @staticmethod
    def f1_score(y, pred):
        return metrics.f1_score(
            Metrics.check_shape(y), Metrics.check_shape(pred)
        )

    @staticmethod
    def auc(y, pred):
        return metrics.roc_auc_score(
            Metrics.check_shape(y, True),
            Metrics.check_shape(pred, True))

    @staticmethod
    def auc(y, pred):
        return metrics.roc_auc_score()