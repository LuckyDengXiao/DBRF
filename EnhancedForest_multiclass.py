# encoding=utf-8

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import numpy as np
import random
import math
from scipy import stats
import time

import ForestUtils
import LogUtils
import AlgorithmUtils

class EnhancedForest():
    
    def __init__(self, train_len, test_len, isLRStacker):
        self.pass_data_x_list = []
        self.pass_data_y_list = []
        self.pass_pred_y_list = []
        self.pass_pred_y_fold_list = []

        self.pass_data_test_list = []
        self.pass_pred_test_list = []
        self.pass_real_test_list = []

        self.train_len = train_len
        self.test_len = test_len
        self.isLRStacker = isLRStacker
        self.loger = LogUtils.LogRecord()

        self.estimator_list = []

    def fit(self, X, y, verbose=True, feval=None, max_depth=None, random_state=1024, 
        min_samples_leaf=100, criterion='gini', num_class=None, kfold=3, n_estimators=50):
        
        kfold = kfold
        # sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.15, random_state=9487)
        sss = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=9487)

        clf_folds = []
        p_all_fold = np.zeros(X.shape[0])
        if num_class:
            p_all_fold_prob = np.zeros((X.shape[0], num_class))
        est_leaf_index = []
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
        
            if verbose:
                print("X_train.shape, y_train.shape:"+str(X_train.shape)+str(y_train.shape))
                print("X_valid.shape, y_valid.shape:"+str(X_valid.shape)+str(y_valid.shape))
        
            clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state, 
                                         min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, n_jobs=-1,
                                         #max_leaf_nodes=100,
    #                                  n_estimators=2, n_jobs=8, oob_score=True, verbose=1, boostrap=False,
                                     criterion=criterion)
            clf = clf.fit(X_train, y_train)
            if verbose:
                print(clf)
                cross_score = cross_val_score(clf, X_valid, y_valid, cv=3)
                print("    cross_score: %.5f" % (cross_score.mean()))
            if not num_class:
                y_pred = clf.predict_proba(X_valid)[:,1]
            else:
                y_pred = clf.predict(X_valid)
                y_pred_prob = clf.predict_proba(X_valid)
                p_all_fold_prob[test_index] += y_pred_prob

            p_all_fold[test_index] += y_pred
            clf_folds.append(clf)

            ## vaild 
            forest_leaf_index = AlgorithmUtils.get_forest_leaf_index(clf, X_valid, y_valid, num_class)
            est_leaf_index.append(forest_leaf_index)

        self.est_leaf_index = est_leaf_index
        self.estimator = clf_folds
        self.estimator_list.append(clf_folds)

        if num_class:
            self.p_all_fold_prob = p_all_fold_prob
            log_model = LogisticRegression()
            log_model.fit(self.p_all_fold_prob, y)
            self.log_model = log_model
        
        # p_all
        if self.isLRStacker:
            p_all = AlgorithmUtils.predict_proba_lr(self.estimator, X, self.log_model, num_class)
        else:
            p_all = AlgorithmUtils.predict_proba(self.estimator, X, num_class)
        self.p_all = p_all
        self.p_all_fold = p_all_fold

        if feval == None:
            print("all data auc", roc_auc_score(y, p_all, labels=[0,1]))
        else:
            print("all data", feval(y, p_all))

        return self.estimator

    def DataSplit(self, X_train, y_train, dropout=0.8, verbose=True):
        if dropout == None:
            pass_data_mask = AlgorithmUtils.get_data_mask_of_ests_by_score(self.estimator, X_train, self.est_leaf_index, verbose)
        elif dropout == -1:
            pass_data_mask = AlgorithmUtils.get_data_mask_of_ests_by_impurity(self.estimator, X_train, self.threshold_imp, verbose)
        else:
            assert (dropout > 0 and dropout < 1)
            pass_data_mask, self.sample_list = AlgorithmUtils.dropout_train(self.estimator, X_train, self.threshold_imp, dropout, verbose)
            # pass_data_mask = self.dropout_train(X_train, dropout, verbose)

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
        print("[p:%d/1:%d|n:%d/1:%d] " % (len(X_train_pass), len(positive_pass), len(X_train_np), len(positive)))#, end="")

        # self.X_train_pass = X_train_pass
        self.X_train_np = X_train_np
        # self.y_train_pass = y_train_pass
        self.y_train_np = y_train_np
        # self.p_all_pass = self.p_all[pass_data_mask]
        self.p_all_np = self.p_all[~pass_data_mask]
        self.pass_data_x_list.append(X_train_pass)
        self.pass_data_y_list.append(y_train_pass)
        self.pass_pred_y_list.append(self.p_all[pass_data_mask])

        # self.p_all_fold_pass = self.p_all_fold[pass_data_mask]
        self.p_all_fold_np = self.p_all_fold[~pass_data_mask]
        self.pass_pred_y_fold_list.append(self.p_all_fold[pass_data_mask])
        # return X_train_pass, y_train_pass, X_train_np, y_train_np
    
    def TestDataSplit(self, X_test, real_y, dropout=0.8, num_class=None, verbose=True):
        """ Split Last Not Passed Test Data
            Return Passed Data Mask & Predict Probability
        Self
        ----
        pass_pred_test_list : list of passed predict_proba of test
        p_test_np : not passed predict_proba of test
        pass_real_test_list : list of passed real test label
        y_test_np : not passed real test label
    
        Parameters
        ----------
        X_test : last not passed X_test
        real_y : last not passed y_test
        
        Returns
        -------
        data_mask : passed data mask of last not passed X_test
        p_test : predict_proba of last not passed X_test
        """
        if dropout == None:
            data_mask = AlgorithmUtils.get_data_mask_of_ests_by_score(self.estimator, X_test, self.est_leaf_index, verbose)
            # data_mask = self.get_data_mask_of_ests_vaild(X_test, verbose)
        elif dropout == -1:
            data_mask = AlgorithmUtils.get_data_mask_of_ests_by_impurity(self.estimator, X_test, self.threshold_imp, verbose)
        else:
            data_mask = AlgorithmUtils.dropout_test(X_test, self.sample_list, self.threshold_imp, dropout, verbose)
            # data_mask = self.dropout_test(X_test, dropout, verbose)
        
        if self.isLRStacker:
            # p_test = self.predict_proba_lr(X_test, num_class)
            p_test = AlgorithmUtils.predict_proba_lr(self.estimator, X_test, self.log_model, num_class)
        else:
            # p_test = self.predict_proba(X_test, num_class)
            p_test = AlgorithmUtils.predict_proba(self.estimator, X_test, num_class)
        
        self.X_test_np = X_test[~data_mask]
        self.pass_data_test_list.append(X_test[data_mask])
        self.pass_pred_test_list.append(p_test[data_mask])
        self.p_test_np = p_test[~data_mask]
        self.pass_real_test_list.append(real_y[data_mask])
        self.y_test_np = real_y[~data_mask]
        return data_mask, p_test 
    
    def TrainModelLayer(self, X, y, X_test=None, all_data_mask=None, y_test=None, real_y=None, verbose=True, feval=None, 
            max_depth=10, random_state=1024, min_samples_leaf=10, criterion='gini', dropout=0.8, 
            isFirst=False, num_class=None, kfold=3, n_estimators=50):
        """ TrainModelLayer
        Parameters
        ----------
        X : last not passed X_train
        y : last not passed y_train
        X_test : all test data X_test
        y_test : all predict test data y_test (change)
        real_y : all test data real_y
        all_data_mask : all test data pass mask
        
        Returns
        -------
        data_mask : passed data mask of last not passed X_test
        all_false_data_index : last not passed X_test
        """
        #训练
        self.fit(X, y, verbose, feval, max_depth, random_state, min_samples_leaf, criterion, num_class, kfold, n_estimators)
        if isFirst: 
            if self.isLRStacker:
                # p_test = self.predict_proba_lr(X_test, num_class)
                p_test = AlgorithmUtils.predict_proba_lr(self.estimator, X_test, self.log_model, num_class)
            else:
                # p_test = self.predict_proba(X_test, num_class)
                p_test = AlgorithmUtils.predict_proba(self.estimator, X_test, num_class)
            self.y_train_np = y
            self.y_test_np = real_y
            self.p_test_np = p_test
            self.p_all_np = self.p_all
            self.p_all_fold_np = self.p_all_fold
            self.X_train_np = X
            self.y_train_np = y
            y_test[:] = p_test[:]
            return self.estimator, np.array([False]*len(X_test)), all_data_mask, p_test
        self.threshold_imp = AlgorithmUtils.get_threshold_of_impurity(self.estimator, self.loger)
        self.DataSplit(X, y, dropout, verbose)

        if type(X_test) == type(None) or type(all_data_mask) == type(None) or type(y_test) == type(None):
            return self.estimator
        
        # X_test_np = X_test[~data_id_mask]
        all_false_data_index = np.where(all_data_mask == False)[0]
        X_test_np = X_test[all_false_data_index]
        y_test_np = real_y[all_false_data_index]
        data_mask, p_test = self.TestDataSplit(X_test_np, y_test_np, dropout, num_class, verbose)
        all_pass_data_index = all_false_data_index[data_mask]
        all_no_pass_data_index = all_false_data_index[~data_mask]
        y_test[all_pass_data_index] = p_test[data_mask]
        y_test[all_no_pass_data_index] = p_test[~data_mask]

        if verbose:
            pass_data_id = data_mask[data_mask==True]
            print("pass test data shape:", len(pass_data_id))
            print("not pass test data shape:", len(X_test_np) - len(pass_data_id))
        # if verbose and type(real_y) != type(None):
        #     if feval == None:
        #         print("pass train data auc", roc_auc_score(self.y_train_pass, self.p_all_pass))
        #         print("pass test data auc", roc_auc_score(real_y[all_pass_data_index], y_test[all_pass_data_index]))
        #     else:
        #         if len(self.p_all_pass) != 0:
        #             print("pass train data", feval(self.y_train_pass, self.p_all_pass))
        #         else:
        #             print("pass train data 0")
        #         if len(y_test[all_pass_data_index]) != 0:
        #             print("pass test data", feval(real_y[all_pass_data_index], y_test[all_pass_data_index]))
        #         else:
        #             print("pass test data 0")
        
        return self.estimator, data_mask, all_false_data_index, p_test
        
    def remove_last_items(self):
        del self.pass_data_x_list[-1]
        del self.pass_pred_y_list[-1]
        del self.pass_data_y_list[-1]
        del self.pass_pred_y_fold_list[-1]

        del self.pass_data_test_list[-1]
        del self.pass_real_test_list[-1]
        del self.pass_pred_test_list[-1]

    ##### ##### ##### ##### 
    ##### get some things
    ##### ##### ##### ##### 

    def getTrainLoss(self, feval=None):
        y_true = self.y_train_np.copy()
        for pass_data_y in self.pass_data_y_list:
            y_true = np.hstack((y_true, pass_data_y))
            #hstack在水平方向上堆叠，也就是把同一行的拼起来，a行b列和a行c列变成a行b+c列
            #pass_data就是需要继续在下一次迭代的时候使用的data
            #not_pass就是分类器已经达成了共识，不需要传递下去了
        y_pred = self.p_all_np.copy()
        for pass_pred_y in self.pass_pred_y_list:
            y_pred = np.hstack((y_pred, pass_pred_y))
        assert y_pred.shape[0] == self.train_len

        if len(y_pred) == 0: return 0,0
        if feval == None:
            return roc_auc_score(y_true, y_pred, labels=[0,1])
        else:
            # return feval(y_true, y_pred, labels=[0,1])
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
            return roc_auc_score(y_true, y_pred, labels=[0,1])
        else:
            # return feval(y_true, y_pred, labels=[0,1])
            return feval(y_true, y_pred)

    def getPassTrainLossNow(self, feval=None):
        if len(self.pass_data_y_list) == 0: return 0,1
        y_true = np.array(self.pass_data_y_list[-1])
        y_pred = np.array(self.pass_pred_y_list[-1])

        if len(y_pred) == 0: return 0,1
        # if len(y_true[y_true==1]) == 0 or len(y_true[y_true==0]) == 0: return 0,0
        if feval == None:
            return roc_auc_score(y_true, y_pred, labels=[0,1])
        else:
            # return feval(y_true, y_pred, labels=[0,1])
            return feval(y_true, y_pred)

    def getVaildLoss(self, feval=None):
        y_true = self.y_train_np.copy()
        for pass_data_y in self.pass_data_y_list:
            y_true = np.hstack((y_true, pass_data_y))
        y_pred = self.p_all_fold_np.copy()
        for pass_pred_y in self.pass_pred_y_fold_list:
            y_pred = np.hstack((y_pred, pass_pred_y))
        assert y_pred.shape[0] == self.train_len

        if len(y_pred) == 0: return 0,0
        if feval == None:
            return roc_auc_score(y_true, y_pred, labels=[0,1])
        else:
            # return feval(y_true, y_pred, labels=[0,1])
            return feval(y_true, y_pred)

    def getPassVaildLoss(self, feval=None):
        y_true = np.array([])
        for pass_data_y in self.pass_data_y_list:
            y_true = np.hstack((y_true, pass_data_y))
        y_pred = np.array([])
        for pass_pred_y in self.pass_pred_y_fold_list:
            y_pred = np.hstack((y_pred, pass_pred_y))

        if len(y_pred) == 0: return 0,0
        if feval == None:
            return roc_auc_score(y_true, y_pred, labels=[0,1])
        else:
            # return feval(y_true, y_pred, labels=[0,1])
            return feval(y_true, y_pred)

    def getPassVaildLossNow(self, feval=None):
        if len(self.pass_data_y_list) == 0: return 0,1
        y_true = np.array(self.pass_data_y_list[-1])
        y_pred = np.array(self.pass_pred_y_fold_list[-1])

        if len(y_pred) == 0: return 0,1
        # if len(y_true[y_true==1]) == 0 or len(y_true[y_true==0]) == 0: return 0,0
        if feval == None:
            return roc_auc_score(y_true, y_pred, labels=[0,1])
        else:
            # return feval(y_true, y_pred, labels=[0,1])
            return feval(y_true, y_pred)

    def getTestLoss(self, feval=None):
        y_true = self.y_test_np.copy()
        for pass_data_test in self.pass_real_test_list:
            y_true = np.hstack((y_true, pass_data_test))
        y_pred = self.p_test_np.copy()
        for pass_pred_test in self.pass_pred_test_list:
            y_pred = np.hstack((y_pred, pass_pred_test))
        assert y_pred.shape[0] == self.test_len

        if len(y_pred) == 0: return 0,0
        if feval == None:
            return roc_auc_score(y_true, y_pred, labels=[0,1])
        else:
            # return feval(y_true, y_pred, labels=[0,1])
            return feval(y_true, y_pred)

    def getTestPred(self):
        y_pred = self.p_test_np.copy()
        for pass_pred_test in self.pass_pred_test_list:
            y_pred = np.hstack((y_pred, pass_pred_test))
        return y_pred

    def getPassTestLoss(self, feval=None):
        y_true = np.array([])
        for pass_data_test in self.pass_real_test_list:
            y_true = np.hstack((y_true, pass_data_test))
        y_pred = np.array([])
        for pass_pred_test in self.pass_pred_test_list:
            y_pred = np.hstack((y_pred, pass_pred_test))

        if len(y_pred) == 0: return 0,0
        if feval == None:
            return roc_auc_score(y_true, y_pred, labels=[0,1])
        else:
            # return feval(y_true, y_pred, labels=[0,1])
            return feval(y_true, y_pred)


    def getPassTestLossNow(self, feval=None):
        if len(self.pass_real_test_list) == 0: return 0,1
        y_true = np.array(self.pass_real_test_list[-1])
        y_pred = np.array(self.pass_pred_test_list[-1])

        if len(y_pred) == 0: return 0,1
        # if len(y_true[y_true==1]) == 0 or len(y_true[y_true==0]) == 0: return 0,0
        if feval == None:
            return roc_auc_score(y_true, y_pred, labels=[0,1])
        else:
            # return feval(y_true, y_pred, labels=[0,1])
            return feval(y_true, y_pred)