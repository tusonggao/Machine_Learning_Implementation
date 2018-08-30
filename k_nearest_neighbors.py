from __future__ import print_function, division

import time
import numpy as np
import pandas as pd

from collections import Counter

class KNN(object):
    def __init__(self, n_neighbors=3):
        self.K = n_neighbors
        return

    def fit(self, train_X, train_y):
        if isinstance(train_X, pd.DataFrame) or isinstance(train_X, pd.Series):
            train_X = train_X.values
        if isinstance(train_y, pd.DataFrame) or isinstance(train_y, pd.Series):
            train_y = train_y.values

        assert len(train_X)==len(train_y), 'train_X must has the same length as train_y'
        self.train_X = train_X
        self.train_y = train_y
        return
    
    def get_result(self, x_vector):
        X_diff = self.train_X - x_vector
        X_diff **= 2
        distance_vector = X_diff.sum(axis=1)
        sorted_idx = np.argsort(np.sqrt(distance_vector))
        y_labels = train_y[sorted_idx]
        y_labels = Counter(y_labels[:self.K])
        y = max(y_labels.keys(), key=lambda x:y_labels[x])
        return y
        
    def predict(self, test_X):
        if isinstance(test_X, pd.DataFrame):
            test_X = test_X.values
        
        if test_X.ndim==1:
            test_X.resize((1, len(test_X)))
        
        assert test_X.shape[1]==self.train_X.shape[1], 'test_X must has the same width as train_X'
            
        y_list = []
        for i in range(test_X.shape[0]):
            x_vector = test_X[i, :]
            y = self.get_result(x_vector)
            y_list.append(y)
        
        return np.array(y_list)
    
# test
if __name__=='__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_mldata
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

#-------------------------------------------------------------

#    train_X = np.array([[1, 1, 1],
#                        [2, 2, 2],
#                        [3, 3, 3],
#                        [4, 4, 4],
#                        [5, 5, 5]])
#    train_y = np.array([0, 1, 1, 1, 1])
#    test_X = np.array([0, 0, 0])
    
#-------------------------------------------------------------
    
    data_path = ('C:/D_Disk/machine_learning_in_action/'
                 'machinelearninginaction-master/Ch02/digits/')
    train_X = np.loadtxt(data_path + 'train_X.txt')
    train_y = np.loadtxt(data_path + 'train_y.txt')
    test_X = np.loadtxt(data_path + 'test_X.txt')
    test_y = np.loadtxt(data_path + 'test_y.txt')
    print('train_X is', train_X.shape)
    print('train_y is', train_y.shape)
    print('test_X is', test_X.shape)
    print('test_y is', test_y.shape)
    
#------------------------------------------------------------------
    
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

#------------------------------------------------------------------
    
    start_t = time.time()
    knn = KNN(n_neighbors=3)
    knn.fit(train_X, train_y)
    y_pred = knn.predict(test_X)
    print('accuacy is ', accuracy_score(test_y, y_pred),
          'cost time: ', time.time()-start_t)

    start_t = time.time()
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_X, train_y) 
    y_pred_sklearn = neigh.predict(test_X)
    print('accuacy is ', accuracy_score(test_y, y_pred_sklearn),
          'cost time: ', time.time()-start_t)
    
    
        
