from decision_tree_cart import DecisionTreeCARTClassifier
import numpy as np
import pandas as pd
import time
from collections import Counter

class RandomForestClassifier_tsg(object):
    def __init__(self, n_estimators = 100, random_state=0):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._built_trees = []

    def fit(self, train_X, train_y):
        if isinstance(train_X, pd.DataFrame) or isinstance(train_X, pd.Series):
            train_X = train_X.values
        if isinstance(train_y, pd.DataFrame) or isinstance(train_y, pd.Series):
            train_y = train_y.values

        assert len(train_X) == len(
            train_y), 'train_X must has the same length as train_y'

        if train_y.ndim == 1:
            train_y = np.reshape(train_y, (-1, 1))

        self.train_X = train_X
        self.train_y = train_y

        np.random.seed(self.random_state)
        for i in range(self.n_estimators):
            select_idx = np.random.choice(len(train_X), len(train_X),
                                          replace=True)
            train_X_selected = train_X[select_idx]
            train_y_selected = train_y[select_idx]
            one_tree = DecisionTreeCARTClassifier()
            one_tree.fit(train_X_selected, train_y_selected)
            self._built_trees.append(one_tree)
        return True

    def _predict_x_vector(self, x_vector):
        outcome_y = []
        for i in range(self.n_estimators):
            y = self._built_trees[i].predict(x_vector)[0]
            outcome_y.append(y)
        y_counter = Counter(outcome_y)
        return max(y_counter.keys(), key=lambda x: y_counter[x])

    def predict(self, test_X):
        if isinstance(test_X, pd.DataFrame):
            test_X = test_X.values

        if test_X.ndim == 1:
            test_X.resize((1, len(test_X)))

        assert test_X.shape[1] == self.train_X.shape[
            1], 'test_X must has the same width as train_X'

        y_list = []
        for i in range(test_X.shape[0]):
            x_vector = test_X[i, :]
            y = self._predict_x_vector(x_vector)
            y_list.append(y)
        return np.array(y_list)


# test
if __name__ == '__main__':
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    # -------------------------------------------------------------
    # 模仿《统计学习方法(李航)》P59表格构造如下数据
    # 特征0 年龄：0 青年  1中年  2 老年
    # 特征1 是否有工作： 0 否 1 是
    # 特征2 有自己的房子: 0 否 1 是
    # 特征3 信贷情况：0 一般 1 好 2 非常好
    # 标签 是否批准贷款： 0 否  1 是
    train_X = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 1, 0, 1],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 1],
                        [1, 1, 1, 1],
                        [1, 0, 1, 2],
                        [1, 0, 1, 2],
                        [2, 0, 1, 2],
                        [2, 0, 1, 1],
                        [2, 1, 0, 1],
                        [2, 1, 0, 2],
                        [2, 0, 0, 0]])
    train_y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    # train_y.resize((len(train_y), 1))
    train_y = np.reshape(train_y, (-1, 1))
    test_X = np.array([0, 1, 0, 0])
    #    test_X = np.array([1, 1, 1, 2])

    print(train_X.shape, train_y.shape, test_X.shape)
    print(np.hstack((train_X, train_y)))

    # -------------------------------------------------------------

    iris = load_iris()
    train_X, test_X, train_y, test_y = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.3,
                                                        random_state=0)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # ------------------------------------------------------------------

    #    SEED = 911
    #    train_df = pd.read_csv('C:/D_Disk/tsg_prog/Digit_Recognizer/mnist_train.csv')
    #    train_df = train_df.sample(20000)
    #    print('train_df.shape is ', train_df.shape)
    #    train_y = train_df['label'].values
    #    train_X = train_df.drop(['label'], axis=1).values
    #    train_X, test_X, train_y, test_y = train_test_split(train_X,
    #                train_y, test_size=0.025, random_state=SEED)
    #
    #    print('after train_test_split, train_X.shape: ', train_X.shape,
    #          'train_y.shape: ', train_y.shape)

    # ------------------------------------------------------------------

    start_t = time.time()
    rf_tsg = RandomForestClassifier_tsg(n_estimators=1000)
    rf_tsg.fit(train_X, train_y)
    y_pred_rf_tsg = rf_tsg.predict(test_X)
    print('y_pred_rf_tsg is ', y_pred_rf_tsg,
          'accuracy_score is ', accuracy_score(test_y, y_pred_rf_tsg),
          'cost time: ', time.time() - start_t)

    start_t = time.time()
    rf = RandomForestClassifier()
    rf.fit(train_X, train_y)
    y_pred_rf = rf.predict(test_X)
    print('y_pred_rf is ', y_pred_rf,
          'accuracy_score is ', accuracy_score(test_y, y_pred_rf),
          'cost time: ', time.time() - start_t)
