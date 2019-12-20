# encoding=utf-8

from sklearn import tree
import time
import datetime
import random
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import xgboost as xgb

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from scipy import stats
from sklearn.linear_model import LogisticRegression
from collections import Counter

def mlp_classifier(X_train, y_train, X_sub, y_sub, feval=metrics.accuracy_score):
    # clf = MLPClassifier(#solver='lbfgs', alpha=1e-5,
    #     max_iter=1000,
    #                  hidden_layer_sizes=(5, 2), random_state=1)
    clf = MLPClassifier(hidden_layer_sizes=(200, 100, 10), max_iter=1000)
    # print(clf)
    clf = clf.fit(X_train, y_train)

    p_train = clf.predict(X_train)
    #p_train = clf.predict_proba(X_train)
    print("train", feval(y_train, p_train))
    p_train = clf.predict(X_sub)
    print("test", feval(y_sub, p_train))
    return clf, p_train

def decision_tree(X_train, y_train, X_sub, y_sub, feval=metrics.accuracy_score):
#     clf = DecisionTreeClassifier(max_depth=None, random_state=1, 
#                                      min_samples_leaf=10, #max_leaf_nodes=100,
# #                                  n_estimators=2, n_jobs=8, oob_score=True, verbose=1, boostrap=False,
#                                  criterion='gini')
    clf = DecisionTreeClassifier()
    # print(clf)
    clf = clf.fit(X_train, y_train)

    p_train = clf.predict(X_train)
    print("train", feval(y_train, p_train))
    p_train = clf.predict(X_sub)
    print("test", feval(y_sub, p_train))
    return clf, p_train

def random_forest(X_train, y_train, X_sub, y_sub, n_estimators=200, feval=metrics.accuracy_score, 
    max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features="auto"):
    # rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, \
    #     min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, max_features=max_features)
        
    ts = time.time()
    tm = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')
    print("start train:", tm)

    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    # print(rf)
    rf = rf.fit(X_train, y_train)

    ts = time.time()
    tm = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')
    print("start test:", tm)

    p_train = rf.predict(X_sub)

    ts = time.time()
    tm = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')
    print("end test:", tm)

    print("test", feval(y_sub, p_train))
    p_train = rf.predict(X_train)
    print("train", feval(y_train, p_train))
    return rf, rf.predict(X_sub)

def xgb_model(X_train, y_train, X_sub, y_sub, num_class, params=None, feval=metrics.accuracy_score):
    '''
    DEBUG: XGB的 watch list 是否参与运算？
    '''
    if params == None:
        params = {
        'objective': 'binary:logistic',
        # 'objective':'multi:softmax',
    #     'objective':'multi:softprob',
        # 'num_class':26
    #     'silent': True,
        
    #     'max_depth': 4,
    #     'eta': 0.020,
    #     'gamma': 0.65,
        
    #     'colsample_bytree': 0.8,
    #     'subsample': 0.6,
        
    #     'num_boost_round' : 700,
    #     'min_child_weight': 10.0,
    #     'max_delta_step': 1.8,
        }
        # params['num_class'] = num_class
    print(params)

    def feval_xgb(preds, dtrain):
        labels = dtrain.get_label()
        # print(preds.shape)
        # acc_score = metrics.accuracy_score(labels, preds)
        # return 'acc', acc_score
        return  feval(labels, preds)

    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=9487)
    # split_obj = tuple(sss.split(X_train, y_train))
    # train_index, test_index = split_obj[0]
    # X_1, X_2 = X_train[train_index], X_train[test_index]
    # y_1, y_2 = y_train[train_index], y_train[test_index]

    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_sub, y_sub)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    print("before train")
    mdl = xgb.train(params, d_train, 
                        num_boost_round=1600, evals=watchlist, early_stopping_rounds=500,
                        feval=feval_xgb,  maximize=True, 
                    verbose_eval=100)
    print(mdl)

    d_train = xgb.DMatrix(X_train)
    p_train = mdl.predict(d_train)
    print(p_train[np.where(p_train>0)[0]])
    print("train", feval(y_train, p_train))
    d_test = xgb.DMatrix(X_sub)
    p_train = mdl.predict(d_test)
    print(p_train[np.where(p_train>0)[0]])
    print("test", feval(y_sub, p_train))
    return mdl


def gdbt_model(X_train, y_train, X_sub, y_sub, n_estimators=200, feval=metrics.accuracy_score):
    gdbt = GradientBoostingClassifier(n_estimators=n_estimators)
    # print(gdbt)
    gdbt = gdbt.fit(X_train, y_train)

    p_train = gdbt.predict(X_train)
    print("train", feval(y_train, p_train))
    p_train = gdbt.predict(X_sub)
    print("test", feval(y_sub, p_train))
    return gdbt, p_train

def predict_proba(clf_folds, X):
    y_preds = np.zeros((X.shape[0], len(clf_folds)))
    for i, clf in enumerate(clf_folds):
        y_pred = clf.predict_proba(X)[:,1]
        y_preds[:, i] = y_pred
    return y_preds.mean(axis=1)

def baseline_model(X_train, y_train, X_sub, y_sub, num_class, feval=metrics.accuracy_score):
    kfold = 3
    X = X_train.copy()
    y = y_train.copy()
    X_test = X_sub.copy()
    real_y = y_sub.copy()
    max_depth = None
    random_state = 1023
    random_state = 1
    min_samples_leaf = 1
    # criterion = 'entropy'#'gini'
    criterion = 'gini'#'gini'
    n_estimators = 50

    sss = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=9487)
    # sss = StratifiedShuffleSplit(n_splits=kfold, random_state=9487)

    tmp1 = np.array(list(Counter(y_train).values()))
    tmp2 = np.array(list(Counter(y_train).keys()))
    index_tmp = np.where(tmp1 == 1)[0]
    for i in index_tmp:
        assert len(index_tmp) > 0
        tmp2_index = i
        data_key = tmp2[tmp2_index]
        tmp_data_index = np.where(y == data_key)[0][0]
        X = np.delete(X, tmp_data_index, axis=0)
        y = np.delete(y, tmp_data_index, axis=0)
        print("--rxz: delete data", tmp_data_index)


    p_all_fold = np.zeros(X.shape[0])
    # p_all_fold_prob = np.zeros((X.shape[0], num_class))
    # p_all_fold_prob_test = np.zeros((X_test.shape[0], num_class))
    p_all_fold_prob = np.zeros((X.shape[0], len(set(y))))
    p_all_fold_prob_test = np.zeros((X_test.shape[0], len(set(y))))
    clf_folds = []

    ###### 
    if len(set(y_train)) != num_class:
        y_index = list(set(y_train))
        y_index.sort()

    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train_bl, X_valid_bl = X[train_index], X[test_index]
        y_train_bl, y_valid_bl = y[train_index], y[test_index]
            
        print("X_train.shape, y_train.shape:"+str(X_train_bl.shape)+str(y_train_bl.shape))
        print("X_valid.shape, y_valid.shape:"+str(X_valid_bl.shape)+str(y_valid_bl.shape))
            
        clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state, 
                                        min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, n_jobs=-1,
                                             #max_leaf_nodes=100,
        #                                  n_estimators=2, n_jobs=8, oob_score=True, verbose=1, boostrap=False,
                                         criterion=criterion)
        clf = clf.fit(X_train_bl, y_train_bl)
        clf_folds.append(clf)
        # print(clf)
        
        cross_score = cross_val_score(clf, X_valid_bl, y_valid_bl, cv=3)
        print("    cross_score: %.5f" % (cross_score.mean()))
        print(real_y)
        print(clf.predict(X_test))
        print("    test score", feval(real_y, clf.predict(X_test)))
        
        
        if not num_class:
            y_pred = clf.predict_proba(X_valid_bl)[:,1]
        else:
            y_pred = clf.predict(X_valid_bl)
            y_pred_prob = clf.predict_proba(X_valid_bl)
            y_pred_prob_test = clf.predict_proba(X_test)

        p_all_fold[test_index] += y_pred
        p_all_fold_prob[test_index] += y_pred_prob
        p_all_fold_prob_test[:, :] += y_pred_prob_test/kfold
        # p_all_fold_prob[:, y_index][test_index] += y_pred_prob
        # p_all_fold_prob_test[:, y_index] += y_pred_prob_test/kfold
    print("train kflod pred:", feval(y, p_all_fold))

    #p_test = predict_proba(clf_folds, X_test)
    p_test = clf.predict(X_test)
    print(p_test[np.where(p_test>0)])
    #print("test data", feval(real_y, p_test, labels=[0,1]))
    print("test data", feval(real_y, p_test))
    #这里报错说多了labels参数

    #### train pred
    y_preds = np.zeros((X.shape[0], len(clf_folds)))
    for i, clf in enumerate(clf_folds):
        if not num_class:
            y_pred = clf.predict(X)
        else:
            y_pred = clf.predict(X)
            #y_pred = clf.predict_proba(X)[:,1]
        y_preds[:, i] = y_pred

    if not num_class:
        m = stats.mode(y_preds, axis=1)
        result = np.array([i[0] for i in m[0]])
    else:
        result = y_preds.mean(axis=1)
    print("train pred", feval(y, result))

    #### test pred
    y_preds = np.zeros((X_test.shape[0], len(clf_folds)))
    for i, clf in enumerate(clf_folds):
        if not num_class:
            y_pred = clf.predict(X_test)
        else:
            y_pred = clf.predict_proba(X_test)[:,1]
        y_preds[:, i] = y_pred

    if not num_class:
        m = stats.mode(y_preds, axis=1)
        result = np.array([i[0] for i in m[0]])
    else:
        result = y_preds.mean(axis=1)
    # print(result)
    print(result[np.where(result>0)])
    print("test pred", feval(real_y, result))

    ### lr pred
    log_model = LogisticRegression()
    print("train lr shape: ", p_all_fold_prob.shape)
    log_model.fit(p_all_fold_prob, y)
    res = log_model.predict(p_all_fold_prob)
    print("lr train", feval(y, res))
    print("test lr shape: ", p_all_fold_prob_test.shape)
    res = log_model.predict(p_all_fold_prob_test)
    print("lr test", feval(real_y, res))
    print("lr coef_", np.mean(log_model.coef_))
    return clf_folds

def baseline_1_flod(X_train, y_train, X_sub, y_sub, num_class, feval=metrics.accuracy_score):
    X = X_train.copy()
    y = y_train.copy()
    X_test = X_sub.copy()
    real_y = y_sub.copy()
    max_depth = None
    random_state = 1024
    min_samples_leaf = 10
    # criterion = 'entropy'#'gini'
    criterion = 'gini'#'gini'
    n_estimators = 200

    X_train_bl, X_valid_bl = X.copy(), X_sub.copy()
    y_train_bl, y_valid_bl = y.copy(), y_sub.copy()
    print("X_train.shape, y_train.shape:"+str(X_train_bl.shape)+str(y_train_bl.shape))
    print("X_valid.shape, y_valid.shape:"+str(X_valid_bl.shape)+str(y_valid_bl.shape))
        
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state, 
                                    min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, n_jobs=-1,
                                         #max_leaf_nodes=100,
    #                                  n_estimators=2, n_jobs=8, oob_score=True, verbose=1, boostrap=False,
                                     criterion=criterion)
    clf = clf.fit(X_train_bl, y_train_bl)
    # print(clf)
    
    cross_score = cross_val_score(clf, X_valid_bl, y_valid_bl, cv=3)
    print("    cross_score: %.5f" % (cross_score.mean()))
    #print("    test score", feval(real_y, clf.predict_proba(X_test)))
    print("    test score", feval(real_y, clf.predict(X_test)))
    
    if not num_class:
        y_pred = clf.predict_proba(X_valid_bl)[:,1]
    else:
        y_pred = clf.predict(X_valid_bl)
    print("train kflod pred:", feval(y_valid_bl, y_pred))

    clf_folds = [clf]
    #### train pred
    y_preds = np.zeros((X.shape[0], len(clf_folds)))
    for i, clf in enumerate(clf_folds):
        if not num_class:
            y_pred = clf.predict_proba(X)[:,1]
        else:
            y_pred = clf.predict(X)
        y_preds[:, i] = y_pred

    if not num_class:
        result = y_preds.mean(axis=1)
    else:
    #     print(y_preds)
        m = stats.mode(y_preds, axis=1)
        result = np.array([i[0] for i in m[0]])
    print("train pred", feval(y, result))

    #### test pred
    y_preds = np.zeros((X_test.shape[0], len(clf_folds)))
    for i, clf in enumerate(clf_folds):
        if not num_class:
            y_pred = clf.predict_proba(X_test)[:,1]
        else:
            y_pred = clf.predict(X_test)
        y_preds[:, i] = y_pred

    if not num_class:
        result = y_preds.mean(axis=1)
    else:
    #     print(y_preds)
        m = stats.mode(y_preds, axis=1)
        result = np.array([i[0] for i in m[0]])
    # print(result)
    print("test pred", feval(real_y, result))
    return clf
