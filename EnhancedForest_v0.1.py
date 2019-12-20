# encoding=utf-8

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

import numpy as np
import random
import math

import ForestUtils
import time

class EnhancedForest():
    
    def __init__(self):
        # self.pass_data_x_list = []
        self.pass_data_y_list = []
        self.pass_pred_y_list = []
        self.pass_pred_test_list = []
        self.pass_data_test_list = []

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
    
        clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state, 
                                     min_samples_leaf=min_samples_leaf, n_estimators=100, n_jobs=-1,
                                     #max_leaf_nodes=100,
#                                  n_estimators=2, n_jobs=8, oob_score=True, verbose=1, boostrap=False,
                                 criterion=criterion)
        clf = clf.fit(X_train, y_train)

        self.estimator = clf
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

        
        p_all = clf.predict_proba(X)
        p_all = [item[1] for item in p_all]
        p_all = np.array(p_all)
        self.p_all = p_all

        if not verbose:
            return clf

        if feval == None:
            print("all data auc", roc_auc_score(y, p_all))
        else:
            print("all data", feval(y, p_all))

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

        return clf
        
        # return clf, X_train, X_valid, y_train, y_valid
  
    # def LayerPerformance(self):
    #     clf = self.estimator
    #     X_train = self.X_train

    #     impurity = clf.tree_.impurity
    #     threshold = clf.tree_.threshold
    #     impurity_index = np.argsort(impurity, axis=0)  

    #     for id in impurity_index:
    #         if threshold[id] != -2: continue
    #         ForestUtils.print_node_info(clf, id)
    #         for item in X_train:
    #             sample_list = [item]
    #             leave_id = clf.apply(sample_list)[0]
    #             if leave_id == id:
    #                 ForestUtils.print_decision_path_ofsample(clf, item)
    #                 break
    #         print()
    
    def get_threshold_of_impurity(self):
        impurity_list = np.array([])
        for estimator in self.estimator.estimators_:
            impurity = estimator.tree_.impurity
            impurity_list = np.hstack((impurity_list, impurity))
        mean_imp = np.mean(impurity_list, axis=0)
        self.threshold_imp = mean_imp

    def get_data_mask_of_tree(self, tree, X_train, verbose=True):
        estimator = tree
        impurity = estimator.tree_.impurity
        pass_node_id_lt = np.where(impurity <= self.threshold_imp)[0]

        node_id_lt = estimator.apply(X_train)
        pass_data_mask = np.isin(node_id_lt, pass_node_id_lt)

        return pass_data_mask

    def get_data_mask_of_forest(self, X_train, verbose=True):
        estimators = self.estimator.estimators_
        last_tree_mask = np.array([True] * len(X_train))
        for index, tree in enumerate(estimators):
            pass_tree_mask = self.get_data_mask_of_tree(tree, X_train, verbose)
            last_tree_mask = last_tree_mask & pass_tree_mask
            if verbose:
                print("%d [%d/%d] " % (index, len(pass_tree_mask[pass_tree_mask==True]), \
                    len(last_tree_mask[last_tree_mask==True])), end="")
        if verbose: print()
        return last_tree_mask

    def dropout_train(self, X_train, dropout=0.8, verbose=True):
        estimators = self.estimator.estimators_
        max_loop = 50
        max_data_mask = np.array([False] * len(X_train))
        for loop in range(max_loop):
            if loop == 0: 
                sample_list = estimators
            elif len(last_tree_mask[last_tree_mask==True]) < 0.01*len(X_train):
                sample_list = random.sample(estimators, math.ceil(dropout*len(sample_list)))
            else:
                if len(last_tree_mask[last_tree_mask==True]) > len(max_data_mask[max_data_mask==True]):
                    max_data_mask = last_tree_mask
                break

            if loop > 0 and len(last_tree_mask[last_tree_mask==True]) > len(max_data_mask[max_data_mask==True]):
                max_data_mask = last_tree_mask
            last_tree_mask = np.array([True] * len(X_train))

            for index, tree in enumerate(sample_list):
                pass_tree_mask = self.get_data_mask_of_tree(tree, X_train, verbose)
                last_tree_mask = last_tree_mask & pass_tree_mask
                if verbose:
                    print("%d [%d/%d] " % (index, len(pass_tree_mask[pass_tree_mask==True]), \
                        len(last_tree_mask[last_tree_mask==True])), end="")
            if verbose: print()
            if verbose:
                print("loop: %d, len sample: %d, len pass: %d" % (loop, len(sample_list), len(last_tree_mask[last_tree_mask==True])))
        if verbose:
            print("loop: %d, len sample: %d, max pass: %d" % (loop, len(sample_list), len(max_data_mask[max_data_mask==True])))
        self.sample_estimators = sample_list
        return max_data_mask

    def dropout_test(self, X_train, dropout=0.8, verbose=True):
        last_tree_mask = np.array([True] * len(X_train))
        for index, tree in enumerate(self.sample_estimators):
            pass_tree_mask = self.get_data_mask_of_tree(tree, X_train, verbose)
            last_tree_mask = last_tree_mask & pass_tree_mask
            if verbose:
                print("%d [%d/%d] " % (index, len(pass_tree_mask[pass_tree_mask==True]), \
                    len(last_tree_mask[last_tree_mask==True])), end="")
        if verbose: print()
        return last_tree_mask

    def DataSplit(self, X_train, y_train, dropout=0.8, verbose=True):
        if dropout != None:
            pass_data_mask = self.dropout_train(X_train, dropout, verbose)
        else:
            pass_data_mask = self.get_data_mask_of_forest(X_train, verbose)

        X_train_pass = X_train[pass_data_mask]
        y_train_pass = y_train[pass_data_mask]
        X_train_np = X_train[~pass_data_mask]
        y_train_np = y_train[~pass_data_mask]
        
        if verbose:
            print("pass data shape:" + str(X_train_pass.shape), y_train_pass[y_train_pass==1].shape)
            print("not pass data shape:" + str(X_train_np.shape), y_train_np[y_train_np==1].shape)
            print("all data shape:" + str(X_train_pass.shape[0]+X_train_np.shape[0]))

        positive = y_train_np[y_train_np==1]
        positive_pass = y_train_pass[y_train_pass==1]
        print("[%d/%d|%d/%d] " % (len(X_train_pass), len(positive_pass), len(X_train_np), len(positive)), end="")

        self.X_train_pass = X_train_pass
        self.X_train_np = X_train_np
        self.y_train_pass = y_train_pass
        self.y_train_np = y_train_np
        self.p_all_pass = self.p_all[pass_data_mask]
        self.p_all_np = self.p_all[~pass_data_mask]
        # self.pass_data_x_list.append(X_train_pass)
        self.pass_data_y_list.append(y_train_pass)
        self.pass_pred_y_list.append(self.p_all_pass)
        # return X_train_pass, y_train_pass, X_train_np, y_train_np
    
    def TestDataSplit(self, X_test, real_y, dropout=0.8, verbose=True):
        if dropout != None:
            data_mask = self.dropout_test(X_test, dropout, verbose)
        else:
            data_mask = self.get_data_mask_of_forest(X_test, verbose)
        
        p_test = self.estimator.predict_proba(X_test)
        p_test = [item[1] for item in p_test]
        p_test = np.array(p_test)

        self.pass_pred_test_list.append(p_test[data_mask])
        self.p_test_np = p_test[~data_mask]
        self.pass_data_test_list.append(real_y[data_mask])
        self.y_test_np = real_y[~data_mask]
        return data_mask, p_test 
    
    def TrainModelLayer(self, X, y, X_test=None, all_data_mask=None, test_y=None, real_y=None, verbose=True, feval=None, 
            max_depth=10, random_state=1024, min_samples_leaf=10, criterion='gini', dropout=0.8):
        
        self.fit(X, y, verbose, feval, max_depth, random_state, min_samples_leaf, criterion)
        self.get_threshold_of_impurity()
        self.DataSplit(X, y, dropout, verbose)

        if type(X_test) == type(None) or type(all_data_mask) == type(None) or type(test_y) == type(None):
            return self.estimator
        
        # X_test_np = X_test[~data_id_mask]
        all_false_data_index = np.where(all_data_mask == False)[0]
        X_test_np = X_test[all_false_data_index]
        y_test_np = real_y[all_false_data_index]
        data_mask, p_test = self.TestDataSplit(X_test_np, y_test_np, dropout, verbose)
        all_pass_data_index = all_false_data_index[data_mask]
        test_y[all_pass_data_index] = p_test[data_mask]

        if verbose:
            pass_data_id = data_mask[data_mask==True]
            print("pass test data shape:", len(pass_data_id))
            print("not pass test data shape:", len(X_test_np) - len(pass_data_id))
        # if verbose and type(real_y) != type(None):
        #     if feval == None:
        #         print("pass train data auc", roc_auc_score(self.y_train_pass, self.p_all_pass))
        #         print("pass test data auc", roc_auc_score(real_y[all_pass_data_index], test_y[all_pass_data_index]))
        #     else:
        #         if len(self.p_all_pass) != 0:
        #             print("pass train data", feval(self.y_train_pass, self.p_all_pass))
        #         else:
        #             print("pass train data 0")
        #         if len(test_y[all_pass_data_index]) != 0:
        #             print("pass test data", feval(real_y[all_pass_data_index], test_y[all_pass_data_index]))
        #         else:
        #             print("pass test data 0")
        
        return self.estimator, data_mask, all_false_data_index, p_test
        
    ##### ##### ##### ##### 
    ##### get some thing
    ##### ##### ##### ##### 

    def getTrainLoss(self, feval=None):
        y_true = self.y_train_np.copy()
        for pass_data_y in self.pass_data_y_list:
            y_true = np.hstack((y_true, pass_data_y))
        y_pred = self.p_all_np.copy()
        for pass_pred_y in self.pass_pred_y_list:
            y_pred = np.hstack((y_pred, pass_pred_y))

        if len(y_pred) == 0: return 0,0
        if feval == None:
            return roc_auc_score(y_true, y_pred)
        else:
            return feval(y_true, y_pred)

    def getPassTrainLoss(self, feval=None):
        y_true = np.array([])
        for pass_data_y in self.pass_data_y_list:
            y_true = np.hstack((y_true, pass_data_y))
        y_pred = np.array([])
        for pass_pred_y in self.pass_pred_y_list:
            y_pred = np.hstack((y_pred, pass_pred_y))

        if len(y_pred) == 0: return 0,0
        if feval == None:
            return roc_auc_score(y_true, y_pred)
        else:
            return feval(y_true, y_pred)

    def getTestLoss(self, feval=None):
        y_true = self.y_test_np.copy()
        for pass_data_test in self.pass_data_test_list:
            y_true = np.hstack((y_true, pass_data_test))
        y_pred = self.p_test_np.copy()
        for pass_pred_test in self.pass_pred_test_list:
            y_pred = np.hstack((y_pred, pass_pred_test))

        if len(y_pred) == 0: return 0,0
        if feval == None:
            return roc_auc_score(y_true, y_pred)
        else:
            return feval(y_true, y_pred)

    def getPassTestLoss(self, feval=None):
        y_true = np.array([])
        for pass_data_test in self.pass_data_test_list:
            y_true = np.hstack((y_true, pass_data_test))
        y_pred = np.array([])
        for pass_pred_test in self.pass_pred_test_list:
            y_pred = np.hstack((y_pred, pass_pred_test))

        if len(y_pred) == 0: return 0,0
        if feval == None:
            return roc_auc_score(y_true, y_pred)
        else:
            return feval(y_true, y_pred)