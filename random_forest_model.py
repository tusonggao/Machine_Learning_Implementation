from decision_tree_cart import DecisionTreeCARTClassifier

class RandomForestClassifier(object):
    def __init__(self):
        print('in init')
        self._built_trees = []

    def fit(self, train_X, train_y, n_estimators = 100):
        return False

    def predict(self, test_X):
        return None

if __name__ == '__main__':
    print('hello world in rf!')