from __future__ import print_function, division

import time
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata

SEED = 911

#mnist = fetch_mldata('MNIST original')
#train_X, test_X, train_y, test_y = train_test_split(mnist.data,
#            mnist.target, test_size=0.2, random_state=SEED)
#print('after train_test_split, train_X.shape: ', train_X.shape,
#      'train_y.shape: ', train_y.shape)


train_df = pd.read_csv('C:/D_Disk/tsg_prog/Digit_Recognizer/mnist_train.csv')
train_df = train_df.sample(3000)
print('train_df.shape is ', train_df.shape)

train_y = train_df['label'].values
train_X = train_df.drop(['label'], axis=1).values

train_X, test_X, train_y, test_y = train_test_split(train_X,
            train_y, test_size=0.2, random_state=SEED)
print('after train_test_split, train_X.shape: ', train_X.shape,
      'train_y.shape: ', train_y.shape)

class KNN():
    def __init__(self, k=5):
        self.K = k
        return

    def fit(self, train_X, train_y):
        assert len(train_X)==len(train_y), 'train_X must has the same length as train_y'
        self.train_X = train_X
        self.train_y = train_y
        return


    def get_result(self, x_vector):
        distance_X = self.train_X - x_vector
        distance_X **= 2
        distance_arr = distance_X.sum(axis=1)
        sorted_idx = np.argsort(distance_arr)
        y_labels = train_y[sorted_idx]
        y_labels = Counter(y_labels[:self.K])
        y = max(y_labels.keys(), key=lambda x:y_labels[x])
        return y
        
    def predict(self, test_X):
        if len(test_X.shape)==1:
            test_X.resize((1, len(test_X)))
        
        assert test_X.shape[1]==self.train_X.shape[1], 'test_X must has the same width as train_X'
            
        y_list = []
        for i in range(test_X.shape[0]):
            x_vector = test_X[i, :]
            y = self.get_result(x_vector)
            y_list.append(y)
        
        return np.array(y_list)
    
if __name__=='__main__':
#    train_X = np.array([[1, 1, 1],
#                        [2, 2, 2],
#                        [3, 3, 3],
#                        [4, 4, 4],
#                        [5, 5, 5]])
#    train_y = np.array([0, 1, 1, 1, 1])
#    test_X = np.array([0, 0, 0])
    
    start_t = time.time()
    knn = KNN(10)
    knn.fit(train_X, train_y)
    y_pred = knn.predict(test_X)
#    print(test_X)
    print('y_pred is ', y_pred)
    print('accuacy is ', (y_pred==test_y).sum()/len(test_y),
          'cost time: ', time.time()-start_t)
    
        
