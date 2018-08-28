from __future__ import print_function, division

import time
import math
import numpy as np
import pandas as pd

from collections import Counter

class DecisionTreeCARTNode(object):
    def __init__(self, fea_idx=None, fea_val=None):
        self.fea_idx = fea_idx  #子节点分裂的feature的维度序列
        self.fea_val = fea_val  #子节点分裂的feature的维度值
        self.child = None
        self.label = None
        return

class DecisionTreeCARTClassifier(object):
    def __init__(self):
        self.root = DecisionTreeCARTNode()
        return
    
    #计算对于划分后的data的基尼值
    def _compute_gini(self, data):
        gini_val = 0.0
        data_num = data.shape[0]
        for y_val in np.unique(data[:, -1]):
            val = (data[:, -1]==y_val).sum()/data_num
            gini_val += val**2
        return 1 - gini_val
    
    #计算对于当前data, 按照fea_idx维、值为fea_val的特征来进行划分后得到的基尼值
    def _compute_split_gini(self, data, fea_idx, fea_val):
        num = data.shape[0]
        equal_num = (data[:, fea_idx]==fea_val).sum()
        non_equal_num = num - equal_num
        gini = 0.0
        gini += equal_num*self._compute_gini(data[data[:, fea_idx]==fea_val])
        gini += non_equal_num*self._compute_gini(data[data[:, fea_idx]!=fea_val])
        return gini/num
    
    def _build_tree_node(self, data):
        #如果所有数据的类别标签都相同 则停止构造子树
        if len(np.unique(data[:, -1]))==1:
            tree_node = DecisionTreeCARTNode()
            tree_node.label = data[1, -1]
            return tree_node
        
        split_fea_idx = None
        split_fea_val = None
        gini_val_min = 999
        for fea_idx in range(data.shape[1]):
            fea_unique_vals = np.unique(data[:, fea_idx])
            if len(fea_unique_vals)==1:
                continue
            for fea_val in fea_unique_vals:
                gini_val = self._compute_split_gini(data, fea_idx, fea_val)
                if gini_val < gini_val_min:
                    gini_val_min = gini_val
                    split_fea_idx = fea_idx
                    split_fea_val = fea_val
        tree_node = DecisionTreeCARTNode(split_fea_idx, split_fea_val)
        equal_data = data[data[:, split_fea_idx]==split_fea_val]
        non_equal_data = data[data[:, split_fea_idx]==split_fea_val]
        tree_node.child = {0: self._build_tree_node(non_equal_data),
                           1: self._build_tree_node(equal_data)}
        return tree_node
        
    def _build_tree(self):
        self.root = self._build_tree_node(data)
        return
    
    def _predict_x_vector(self, x_vector):
        node = self.root
        while node.child!=None:
            fea_val = x_vector[node.fea_idx]
            if fea_val==self.fea_val:
                node = node.child[1]
            else:
                node = node.child[0]
        return node.label
    
    def fit(self, train_X, train_y):
        if isinstance(train_X, pd.DataFrame) or isinstance(train_X, pd.Series):
            train_X = train_X.values
        if isinstance(train_y, pd.DataFrame) or isinstance(train_y, pd.Series):
            train_y = train_y.values

        assert len(train_X)==len(train_y), 'train_X must has the same length as train_y'
        self.train_X = train_X
        self.train_y = train_y
        data = []
        
        self._build_tree()
        
        return
    
    def predict(self, test_X):
        if isinstance(test_X, pd.DataFrame):
            test_X = test_X.values
        
        if len(test_X.shape)==1:
            test_X.resize((1, len(test_X)))
        
        assert test_X.shape[1]==self.train_X.shape[1], 'test_X must has the same width as train_X'
            
        y_list = []
        for i in range(test_X.shape[0]):
            x_vector = test_X[i, :]
            y = self._predict_x_vector(x_vector)
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
    
#    data_path = ('C:/D_Disk/machine_learning_in_action/'
#                 'machinelearninginaction-master/Ch02/digits/')
#    train_X = np.loadtxt(data_path + 'train_X.txt')
#    train_y = np.loadtxt(data_path + 'train_y.txt')
#    test_X = np.loadtxt(data_path + 'test_X.txt')
#    test_y = np.loadtxt(data_path + 'test_y.txt')
#    print('train_X is', train_X.shape)
#    print('train_y is', train_y.shape)
#    print('test_X is', test_X.shape)
#    print('test_y is', test_y.shape)
    
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
    
#    start_t = time.time()
#    knn = KNN(n_neighbors=3)
#    knn.fit(train_X, train_y)
#    y_pred = knn.predict(test_X)
#    print('accuacy is ', accuracy_score(test_y, y_pred),
#          'cost time: ', time.time()-start_t)
#
#    start_t = time.time()
#    neigh = KNeighborsClassifier(n_neighbors=3)
#    neigh.fit(train_X, train_y) 
#    y_pred_sklearn = neigh.predict(test_X)
#    print('accuacy is ', accuracy_score(test_y, y_pred_sklearn),
#          'cost time: ', time.time()-start_t)