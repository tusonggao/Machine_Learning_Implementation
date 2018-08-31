#from .decision_tree_cart import DecisionTreeCARTClassifier
import numpy as np
from collections import Counter

class RandomForestClassifier(object):
    def __init__(self):
        print('in init')
        self.n_estimators = 0
        self._built_trees = []

    def bootstrap_select(self, train_X, train_y):
        #return select_X, select_y
        return True

    def fit(self, train_X, train_y, n_estimators = 100):
        self.n_estimators = n_estimators
        train_X = np.random.randint(0, len(train_X)-1, len(train_X))
        for i in range(n_estimators):
            select_idx = np.random.choice(len(train_X), len(train_X),
                                          replace=True)
            train_X_selected = train_X[select_idx]
            train_y_selected = train_y[select_idx]
            one_tree = DecisionTreeCARTClassifier()
            one_tree.fit(train_X_selected, train_y_selected)
            selef._built_trees.append(one_tree)

        return True

    def predict(self, test_X):
        outcome_y = []

        for i in range(outcome_y)
        return None

if __name__ == '__main__':
    print('hello world in rf!')


    counter_selected = Counter(selected)

    print('len of counter_selected is ', len(counter_selected))
