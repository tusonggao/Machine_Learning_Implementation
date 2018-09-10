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
    def __init__(self, leaf_instance_num=2):
        self.root = DecisionTreeCARTNode()
        self.leaf_instance_num = leaf_instance_num
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
        #如果所有数据的类别标签都相同 或者 超过了叶子节点最小样本个数 则停止构造子树
        if (len(np.unique(data[:, -1]))==1 or
                self.leaf_instance_num >= len(data)):
            tree_node = DecisionTreeCARTNode()
            tree_node.label = data[0, -1]
            return tree_node

        split_fea_idx = None
        split_fea_val = None
        gini_val_min = 999
        for fea_idx in range(data.shape[1]-1):
            fea_unique_vals = np.unique(data[:, fea_idx])
            if len(fea_unique_vals)==1: #特征值都相同，无需分裂子节点
                continue
            for fea_val in fea_unique_vals:
                gini_val = self._compute_split_gini(data, fea_idx, fea_val)
                if gini_val < gini_val_min:
                    gini_val_min = gini_val
                    split_fea_idx = fea_idx
                    split_fea_val = fea_val
        tree_node = DecisionTreeCARTNode(split_fea_idx, split_fea_val)
        
        if gini_val_min!=999: #发生子节点分裂
            equal_data = data[data[:, split_fea_idx]==split_fea_val]
            non_equal_data = data[data[:, split_fea_idx]!=split_fea_val]
            tree_node.child = {0: self._build_tree_node(non_equal_data),
                               1: self._build_tree_node(equal_data)}
        else:  #所有值相等 且不再刻意分裂  则将出现次数最多的标签作为最后标签
            y_counter = Counter(data[:, -1])
            tree_node.label = max(y_counter.keys(), key=lambda x:y_counter[x])
            print('not split with two label')
        return tree_node
        
    def _build_tree(self, data):
        self.root = self._build_tree_node(data)
        return
    
    def _predict_x_vector(self, x_vector):
        node = self.root
        while node.child!=None:
            fea_val = x_vector[node.fea_idx]
            if fea_val==node.fea_val:
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

        if train_y.ndim == 1:
            train_y = np.reshape(train_y, (-1, 1))
        
        self.train_X = train_X
        self.train_y = train_y
        self._build_tree(np.hstack((train_X, train_y)))
        return
    
    def predict(self, test_X):
        if isinstance(test_X, pd.DataFrame):
            test_X = test_X.values
        
        if test_X.ndim==1:
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
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_iris

#-------------------------------------------------------------
#模仿《统计学习方法(李航)》P59表格构造如下数据
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

#-------------------------------------------------------------

    # iris = load_iris()
    # train_X, test_X, train_y, test_y = train_test_split(iris.data,
    #         iris.target, test_size=0.5, random_state=0)
    #
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # clf = DecisionTreeClassifier(random_state=0)
    # scores = cross_val_score(clf, iris.data, iris.target, cv=10)
    # print(scores)

#------------------------------------------------------------------

   SEED = 911
   train_df = pd.read_csv('C:/D_Disk/tsg_prog/Digit_Recognizer/mnist_train.csv')
   train_df = train_df.sample(20000)
   print('train_df.shape is ', train_df.shape)
   train_y = train_df['label'].values
   train_X = train_df.drop(['label'], axis=1).values
   train_X, test_X, train_y, test_y = train_test_split(train_X,
               train_y, test_size=0.025, random_state=SEED)

   print('after train_test_split, train_X.shape: ', train_X.shape,
         'train_y.shape: ', train_y.shape)

#------------------------------------------------------------------

    start_t = time.time()
    treeCART = DecisionTreeCARTClassifier()
    treeCART.fit(train_X, train_y)
    y_pred_cart = treeCART.predict(test_X)
    print('y_pred_cart is ', y_pred_cart,
          'accuracy_score is ', accuracy_score(test_y, y_pred_cart),
          'cost time: ', time.time()-start_t)

    start_t = time.time()
    treeSklearn = DecisionTreeClassifier()
    treeSklearn.fit(train_X, train_y)
    y_pred_sklearn = treeSklearn.predict(test_X)
    print('y_pred_sklearn is ', y_pred_sklearn,
          'accuracy_score is ', accuracy_score(test_y, y_pred_sklearn),
          'cost time: ', time.time()-start_t)

    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    np.hstack((a, b))




#------------------------------------------------------------------

    # start_t = time.time()
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
#     print('accuacy is ', accuracy_score(test_y, y_pred_sklearn),
#          'cost time: ', time.time()-start_t)