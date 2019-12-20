# encoding=utf-8

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

import numpy as np

import ForestUtils
import time

class EnhancedDTree():
    
    def __init__(self,
                 criterion=None):
        self.criterion = criterion

    def fit(self, X, y, verbose=True, feval=None, 
            max_depth=None, random_state=1024, min_samples_leaf=100, criterion='gini'):
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=9487)
        split_obj = tuple(sss.split(X, y))
        train_index, test_index = split_obj[0]
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        
        if verbose:
            print("X_train.shape, y_train.shape:"+str(X_train.shape)+str(y_train.shape))
            print("X_valid.shape, y_valid.shape:"+str(X_valid.shape)+str(y_valid.shape))
    
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, 
                                     min_samples_leaf=min_samples_leaf, #max_leaf_nodes=100,
#                                  n_estimators=2, n_jobs=8, oob_score=True, verbose=1, boostrap=False,
                                 criterion=criterion)
        clf = clf.fit(X_train, y_train)

        if not verbose:
            return clf, X_train, X_valid, y_train, y_valid
        
        p_train = clf.predict_proba(X)
        p_train = [item[1] for item in p_train]
        p_train = np.array(p_train)
        if feval == None:
            print("all data auc", roc_auc_score(y, p_train))
        else:
            print("all data", feval(y, p_train))

        p_train = clf.predict_proba(X_train)
        p_train = [item[1] for item in p_train]
        p_train = np.array(p_train)
        if feval == None:
            print("train data auc", roc_auc_score(y_train, p_train))
        else:
            print("train data", feval(y_train, p_train))

        p_valid = clf.predict_proba(X_valid)
        p_valid = [item[1] for item in p_valid]
        p_valid = np.array(p_valid)
        if feval == None:
            print("valid data auc", roc_auc_score(y_valid, p_valid))
        else:
            print("valid data", feval(y_valid, p_valid))
        
        self.estimator = clf
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        # return clf, X_train, X_valid, y_train, y_valid
  
    def LayerPerformance(self):
        clf = self.estimator
        X_train = self.X_train

        impurity = clf.tree_.impurity
        threshold = clf.tree_.threshold
        impurity_index = np.argsort(impurity, axis=0)  

        for id in impurity_index:
            if threshold[id] != -2: continue
            ForestUtils.print_node_info(clf, id)
            for item in X_train:
                sample_list = [item]
                leave_id = clf.apply(sample_list)[0]
                if leave_id == id:
                    ForestUtils.print_decision_path_ofsample(clf, item)
                    break
            print()
    
    def DataSplit(self, X_train, y_train, verbose=True):
        estimator = self.estimator
        # X_train = self.X_train
        # y_train = self.y_train

        data_id_list = np.array(list(range(len(X_train))))
        
        # S0 对于一个决策树返回所有叶子节点的gini排序计算平均
        impurity = estimator.tree_.impurity
        mean_imp = np.mean(impurity, axis=0)
        #########################
        min_imp = np.min(impurity)
        # mean_imp = (min_imp+mean_imp)/2
        mean_imp = min_imp
        #########################
        if verbose:
            print("mean of all impurity:" + str(mean_imp))

        # S1 通过gini获取相关节点id
        pass_node_id_lt = np.where(impurity <= mean_imp)[0]
        if verbose:
            print("pass node id shape:" + str(pass_node_id_lt.shape))

        # S2 通过node id获取相关数据
        node_id_lt = estimator.apply(X_train)
        node_id_mask = np.isin(node_id_lt, pass_node_id_lt)

        # S3 通过pass data id分割数据
        pass_data_mask = node_id_mask
        X_train_pass = X_train[pass_data_mask]
        y_train_pass = y_train[pass_data_mask]
        X_train_np = X_train[~pass_data_mask]
        y_train_np = y_train[~pass_data_mask]
        
        if verbose:
            print("pass data shape:" + str(X_train_pass.shape), y_train_pass[y_train_pass==1].shape)
            print("not pass data shape:" + str(X_train_np.shape), X_train_np[X_train_np==1].shape)
            print("all data shape:" + str(X_train_pass.shape[0]+X_train_np.shape[0]))

        self.X_train_pass = X_train_pass
        self.y_train_pass = y_train_pass
        self.X_train_np = X_train_np
        self.y_train_np = y_train_np
        # return X_train_pass, y_train_pass, X_train_np, y_train_np
    
    def TestDataSplit(self, X_test, verbose=True):
        estimator = self.estimator
        # S0 对于一个决策树返回所有叶子节点的gini排序计算平均
        impurity = estimator.tree_.impurity
        mean_imp = np.mean(impurity, axis=0)
        #########################
        # min_imp = np.min(impurity)
        # mean_imp = (min_imp+mean_imp)/2
        # mean_imp = min_imp
        #########################
        if verbose:
            print("mean of all impurity:" + str(mean_imp))

        # S1 通过gini获取相关节点id
        pass_node_id_lt = np.where(impurity <= mean_imp)[0]
        if verbose:
            print("pass node id shape:" + str(pass_node_id_lt.shape))

        # S2 通过node id获取相关数据
        data_node_id_lt = estimator.apply(X_test)
        data_mask = np.isin(data_node_id_lt, pass_node_id_lt)

        # S3 输出预测结果
        p_test = estimator.predict_proba(X_test)
        p_test = [item[1] for item in p_test]
        p_test = np.array(p_test)

        return data_mask, p_test 
    
    def TrainModelLayer(self, X, y, X_test=None, all_data_mask=None, test_y=None, real_y=None, verbose=True, feval=None, 
            max_depth=10, random_state=1024, min_samples_leaf=100, criterion='gini'):
        
        self.fit(X, y, verbose, feval, max_depth, random_state, min_samples_leaf, criterion)
        self.DataSplit(X, y, verbose)
        
        if type(X_test) == type(None) or type(all_data_mask) == type(None) or type(test_y) == type(None):
            return self.estimator
        
        # X_test_np = X_test[~data_id_mask]
        all_false_data_index = np.where(all_data_mask == False)[0]
        X_test_np = X_test[all_false_data_index]
        data_mask, p_test = self.TestDataSplit(X_test_np)
        all_pass_data_index = all_false_data_index[data_mask]
        test_y[all_pass_data_index] = p_test[data_mask]

        if verbose:
            pass_data_id = data_mask[data_mask==True]
            print("pass test data shape:", len(pass_data_id))
            print("not pass test data shape:", len(X_test_np) - len(pass_data_id))
        if type(real_y) != type(None):
            if feval == None:
                print("pass test data auc", roc_auc_score(real_y[all_pass_data_index], test_y[all_pass_data_index]))
            else:
                print("pass test data", feval(real_y[all_pass_data_index], test_y[all_pass_data_index]))

        
        return self.estimator, data_mask, all_false_data_index, p_test
        