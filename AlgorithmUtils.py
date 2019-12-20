# encoding=utf-8

import numpy as np
from scipy import stats
import math
from sklearn import tree as sklearn_tree
import graphviz

def predict_proba(estimators, X, num_class=None):
    y_preds = np.zeros((X.shape[0], len(estimators)))
    for i, clf in enumerate(estimators):
        if not num_class:
            y_pred = clf.predict_proba(X)[:,1]
        else:
            y_pred = clf.predict(X)
        y_preds[:, i] = y_pred

    if not num_class:
        return y_preds.mean(axis=1)
    else:
        m = stats.mode(y_preds, axis=1)
        return np.array([i[0] for i in m[0]])

def predict_proba_lr(estimators, X, log_model, num_class=None):
    y_preds = np.zeros((X.shape[0], num_class))
    for i, clf in enumerate(estimators):
        y_pred = clf.predict_proba(X)
        y_preds[:, :] += y_pred/len(estimators)
    return log_model.predict(y_preds)

def get_forest_leaf_index(clf, X_valid, y_valid, num_class):
    if num_class == None: num_class = 2

    forest_leaf_index = []
    for index, tree in enumerate(clf):
        # max dimension
        # max_dim = 0
        # get all data node id list
        node_id_lt_a = tree.apply(X_valid)
        node_id_cnt_a = np.bincount(node_id_lt_a)
        max_dim = np.max(node_id_lt_a) + 1
        # if tmp_dim_a > max_dim: max_dim = tmp_dim_a

        node_id_cnt_class = []
        for i in range(num_class):
            if len(X_valid[y_valid==i]) == 0: continue
            node_id_lt_tmp = tree.apply(X_valid[y_valid==i])
            node_id_cnt_tmp = np.bincount(node_id_lt_tmp)
            tmp_dim = np.max(node_id_lt_tmp) + 1
            if tmp_dim < max_dim:
                diff = max_dim - tmp_dim
                node_id_cnt_tmp = np.append(node_id_cnt_tmp, [0]*diff)
            # assert
            assert not any(~np.isfinite(node_id_cnt_tmp))
            node_id_cnt_class.append(node_id_cnt_tmp)
        node_id_cnt_class = np.array(node_id_cnt_class)
        
        # node_id_most
        node_id_cnt_m = np.max(node_id_cnt_class, axis=0)
        # node_id_cnt_p_n = np.vstack((node_id_cnt_p, node_id_cnt_n))
        node_id_argmax = np.argmax(node_id_cnt_class, axis=0)
        node_id_y_prob = np.array([1] * max_dim)
        for i in range(num_class):
            node_id_y_prob[node_id_argmax == i] = node_id_y_prob[node_id_argmax == i] * len(X_valid[y_valid==i])/len(X_valid)
        # node_id_argmin = np.argmin(node_id_cnt_p_n, axis=0)
        # node_id_argmax = node_id_argmax * len(X_valid[y_valid==0])/len(X_valid)
        # node_id_argmin = node_id_argmin * len(X_valid[y_valid==1])/len(X_valid)
        # node_id_y_prob = node_id_argmax + node_id_argmin

        # assert
        assert node_id_cnt_m.shape == node_id_cnt_a.shape
        # node_id_freq & node_id_count
        node_id_freq = node_id_cnt_m/node_id_cnt_a
        node_id_lift = node_id_freq/node_id_y_prob
        node_id_count = node_id_cnt_m/len(X_valid)
        # print(node_id_count, node_id_freq, node_id_count*node_id_freq)
        # node_id_score = node_id_count*node_id_freq
        node_id_score = 2*node_id_freq*node_id_count/(node_id_freq+node_id_count)
        # node_id_score = 2*node_id_freq*node_id_lift/(node_id_freq+node_id_lift)
        # node_id_score = node_id_lift
        # node_id_score = node_id_freq


        # get positive data node id list
        # node_id_lt_p = tree.apply(X_valid[y_valid==1])
        # node_id_cnt_p = np.bincount(node_id_lt_p)
        # tmp_dim_p = np.max(node_id_lt_p)
        # if tmp_dim_p > max_dim: max_dim = tmp_dim_p
        # # get negative data node id list
        # node_id_lt_n = tree.apply(X_valid[y_valid==0])
        # node_id_cnt_n = np.bincount(node_id_lt_n)
        # tmp_dim_n = np.max(node_id_lt_n)
        # if tmp_dim_n > max_dim: max_dim = tmp_dim_n
        # # sync dimension
        # if tmp_dim_a < max_dim:
        #     diff = max_dim - tmp_dim_a
        #     node_id_cnt_a = np.append(node_id_cnt_a, [0]*diff)
        # if tmp_dim_p < max_dim:
        #     diff = max_dim - tmp_dim_p
        #     node_id_cnt_p = np.append(node_id_cnt_p, [0]*diff)
        # if tmp_dim_n < max_dim:
        #     diff = max_dim - tmp_dim_n
        #     node_id_cnt_n = np.append(node_id_cnt_n, [0]*diff)
        # # assert
        # assert not any(~np.isfinite(node_id_cnt_a))
        # assert not any(~np.isfinite(node_id_cnt_p))
        # assert not any(~np.isfinite(node_id_cnt_n))

        # node_id_most
        # node_id_cnt_m = np.maximum(node_id_cnt_p, node_id_cnt_n)
        # node_id_cnt_p_n = np.vstack((node_id_cnt_p, node_id_cnt_n))
        # node_id_argmax = np.argmax(node_id_cnt_p_n, axis=0)
        # node_id_argmin = np.argmin(node_id_cnt_p_n, axis=0)
        # node_id_argmax = node_id_argmax * len(X_valid[y_valid==0])/len(X_valid)
        # node_id_argmin = node_id_argmin * len(X_valid[y_valid==1])/len(X_valid)
        # node_id_y_prob = node_id_argmax + node_id_argmin

        # # assert
        # assert node_id_cnt_m.shape == node_id_cnt_a.shape
        # # node_id_freq & node_id_count
        # node_id_freq = node_id_cnt_m/node_id_cnt_a
        # node_id_lift = node_id_freq/node_id_y_prob
        # node_id_count = node_id_cnt_m/len(X_valid)
        # print(node_id_count, node_id_freq, node_id_count*node_id_freq)
        # node_id_score = node_id_count*node_id_freq
        # node_id_score = 2*node_id_freq*node_id_count/(node_id_freq+node_id_count)
        node_id_score = 2*node_id_freq*node_id_lift/(node_id_freq+node_id_lift)
        # node_id_score = node_id_freq

        # !!! FIXME: Maybe bugs
        node_id_score[np.isnan(node_id_score)] = 0
        node_id_score[np.isinf(node_id_score)] = 0
        assert not any(~np.isfinite(node_id_score))

        impurity_index = np.argsort(node_id_score, axis=0)[::-1]  
        impurity_sort = np.sort(node_id_score)[::-1]

        threshold_imp = np.mean(node_id_score[node_id_score>0], axis=0)
        # print(impurity_sort)
        now_tree_impurity_index = impurity_index[impurity_sort>threshold_imp]
        now_tree_impurity_sort = impurity_sort[impurity_sort>threshold_imp]
        # print(now_tree_impurity_index.shape, node_id_score.shape)
        # print(now_tree_impurity_index)
        # print(now_tree_impurity_sort)
        forest_leaf_index.append(now_tree_impurity_index)
    return forest_leaf_index

def _get_leaf_node_impurity(estimator):
    a = np.where(estimator.tree_.children_left == -1)[0]
    b = np.where(estimator.tree_.children_right == -1)[0]
    c = np.where(estimator.tree_.feature == -2)[0]
    d = np.where(estimator.tree_.threshold == -2)[0]
    assert (a == b).all() and (a == c).all() and (a == d).all()
    return estimator.tree_.impurity[a]
    
def get_threshold_of_impurity(estimators, loger):
    impurity_list = np.array([])
    for i, est in enumerate(estimators):
        for j, tree in enumerate(est):
            # impurity = estimator.tree_.impurity
            impurity = _get_leaf_node_impurity(tree)
            impurity_list = np.hstack((impurity_list, impurity))

            # dot_data = sklearn_tree.export_graphviz(tree, out_file=None, 
            #              class_names=['neg', 'pos'],  
            #              filled=True, rounded=True,  
            #              special_characters=True) 
            # graph = graphviz.Source(dot_data)  
            # graph.render("./logs/trees/"+str(i*len(estimators)+j)) 
    mean_imp = np.mean(impurity_list, axis=0)

    #loger.log_array(impurity_list, "impurity_list")
    #loger.log_value(mean_imp, "mean_imp")
    # self.threshold_imp = mean_imp
    return mean_imp

def _get_data_mask_of_tree(tree, X_train, threshold_imp, verbose):
    estimator = tree
    # impurity = estimator.tree_.impurity
    impurity = _get_leaf_node_impurity(estimator)
    pass_node_id_lt = np.where(impurity <= threshold_imp)[0]
    # self.loger.log_array(pass_node_id_lt, "pass_node_id_lt")
    # self.loger.log_value(pass_node_id_lt.shape, "shape")
    # print("rxz test", pass_node_id_lt)
    # print("rxz test", pass_node_id_lt.shape)

    node_id_lt = estimator.apply(X_train)
    pass_data_mask = np.isin(node_id_lt, pass_node_id_lt)
    # test_pass_data_mask = np.isin(np.sort(np.unique(node_id_lt)), pass_node_id_lt)
    # self.loger.log_array(node_id_lt, "node_id_lt")
    # self.loger.log_array(pass_data_mask, "pass_data_mask")
    # self.loger.log_array(np.sort(np.unique(node_id_lt), axis=0) , "node_id_lt")
    # self.loger.log_array(test_pass_data_mask, "test_pass_data_mask")
    # self.loger.log_value(np.unique(node_id_lt).shape, "unique_node_id_lt.shape")
    # self.loger.log_value(X_train.shape, "X_train.shape")
    # self.loger.log_value(pass_data_mask[pass_data_mask==True].shape, "X_train_pass.shape")
    # self.loger.log_value(pass_data_mask[pass_data_mask==False].shape, "X_train_npass.shape")

    return pass_data_mask

# def _get_data_mask_of_forest(forest, X_train, threshold_imp, verbose):
#     last_tree_mask = np.array([True] * len(X_train))
#     for index, tree in enumerate(forest):
#         pass_tree_mask = _get_data_mask_of_tree(tree, X_train, threshold_imp, verbose)
#         last_tree_mask = last_tree_mask & pass_tree_mask
#         if verbose:
#             print("%d [%d/%d] " % (index, len(pass_tree_mask[pass_tree_mask==True]), \
#                 len(last_tree_mask[last_tree_mask==True])), end="")
#     if verbose: print()
#     return last_tree_mask

# def get_data_mask_of_ests(estimators, X_train, threshold_imp, verbose=True):
#     # estimators = self.estimator
#     last_forest_mask = np.array([True] * len(X_train))
#     for index, forest in enumerate(estimators):
#         pass_forest_mask = _get_data_mask_of_forest(forest, X_train, verbose)
#         last_forest_mask = last_forest_mask & pass_forest_mask
#         if verbose:
#             print("%d [%d/%d] " % (index, len(pass_forest_mask[pass_forest_mask==True]), \
#                 len(last_forest_mask[last_forest_mask==True])), end="")
#     if verbose: print()
#     return last_forest_mask

def get_data_mask_of_ests_by_impurity(estimators, X_train, threshold_imp, verbose=True):
    # estimators = self.estimator
    last_forest_mask = np.array([True] * len(X_train))
    for index, forest in enumerate(estimators):
        for i_tree, tree in enumerate(forest):
            # impurity = _get_leaf_node_impurity(tree)
            # pass_node_id_lt = np.where(impurity <= threshold_imp)[0]
            # node_id_lt = tree.apply(X_train)
            # pass_data_mask = np.isin(node_id_lt, pass_node_id_lt)
            pass_data_mask = _get_data_mask_of_tree(tree, X_train, threshold_imp, verbose)
            last_forest_mask = last_forest_mask & pass_data_mask
            if verbose:
                print("%d leaf-num:%d[now:%d/all:%d] " % (index, len(tree_leaf_index[i_tree]), \
                    len(pass_data_mask[pass_data_mask==True]), \
                    len(last_forest_mask[last_forest_mask==True])), end="")
        if verbose: print()
    return last_forest_mask

def get_data_mask_of_ests_by_score(estimators, X_train, est_leaf_index, verbose=True):
    # estimators = self.estimator
    last_forest_mask = np.array([True] * len(X_train))
    for index, forest in enumerate(estimators):
        tree_leaf_index = est_leaf_index[index]
        for i_tree, tree in enumerate(forest):
            node_id_lt = tree.apply(X_train)
            pass_data_mask = np.isin(node_id_lt, tree_leaf_index[i_tree])
            last_forest_mask = last_forest_mask & pass_data_mask
            if verbose:
                print("%d leaf-num:%d[now:%d/all:%d] " % (index, len(tree_leaf_index[i_tree]), \
                    len(pass_data_mask[pass_data_mask==True]), \
                    len(last_forest_mask[last_forest_mask==True])), end="")
        if verbose: print()
    return last_forest_mask

def _get_all_trees(estimator):
    trees = []
    for i, clf in enumerate(estimator):
        clf_trees = clf.estimators_
        trees.extend(clf_trees)
    return np.array(trees)

def _get_trees_sort_index(trees):
    impuritys = []
    for tree in trees:
        # impurity = tree.tree_.impurity
        impurity = _get_leaf_node_impurity(tree)
        imp_mean = np.mean(impurity, axis=0)
        impuritys.append(imp_mean)

    trees_sort_index = np.argsort(impuritys, axis=0) 
    return trees_sort_index

def _get_dropout_trees(trees, last_len, trees_sort_index, dropout):
    max_len = math.ceil(dropout*last_len)
    dropout_index = trees_sort_index[:max_len]
    return trees[dropout_index]

def dropout_train(estimators, X_train, threshold_imp, dropout=0.8, verbose=True):
    # estimators = self.estimator.estimators_
    estimators = _get_all_trees(estimators)
    trees_sort_index = _get_trees_sort_index(estimators)
    max_loop = 50
    max_data_mask = np.array([False] * len(X_train))
    for loop in range(max_loop):
        if loop == 0: 
            sample_list = estimators
        elif len(last_tree_mask[last_tree_mask==True]) < 0.01*len(X_train):
            # sample_list = random.sample(estimators, math.ceil(dropout*len(sample_list)))
            sample_list = _get_dropout_trees(estimators, len(sample_list), trees_sort_index, dropout)
        else:
            if len(last_tree_mask[last_tree_mask==True]) > len(max_data_mask[max_data_mask==True]):
                max_data_mask = last_tree_mask
            break

        if loop > 0 and len(last_tree_mask[last_tree_mask==True]) > len(max_data_mask[max_data_mask==True]):
            max_data_mask = last_tree_mask
        last_tree_mask = np.array([True] * len(X_train))

        for index, tree in enumerate(sample_list):
            pass_tree_mask = _get_data_mask_of_tree(tree, X_train, threshold_imp, verbose)
            last_tree_mask = last_tree_mask & pass_tree_mask
            if verbose:
                print("%d [%d/%d] " % (index, len(pass_tree_mask[pass_tree_mask==True]), \
                    len(last_tree_mask[last_tree_mask==True])), end="")
        if verbose: print()
        if verbose:
            print("loop: %d, len sample: %d, len pass: %d" % (loop, len(sample_list), len(last_tree_mask[last_tree_mask==True])))
    if verbose:
        print("loop: %d, len sample: %d, max pass: %d" % (loop, len(sample_list), len(max_data_mask[max_data_mask==True])))
    # self.sample_trees = sample_list
    return max_data_mask, sample_list

def dropout_test(X_train, sample_trees, threshold_imp, dropout=0.8, verbose=True):
    last_tree_mask = np.array([True] * len(X_train))
    for index, tree in enumerate(sample_trees):
        pass_tree_mask = _get_data_mask_of_tree(tree, X_train, threshold_imp, verbose)
        last_tree_mask = last_tree_mask & pass_tree_mask
        if verbose:
            print("%d [%d/%d] " % (index, len(pass_tree_mask[pass_tree_mask==True]), \
                len(last_tree_mask[last_tree_mask==True])), end="")
    if verbose: print()
    return last_tree_mask