# encoding=utf-8
import numpy as np

def print_decision_path_ofsample(estimator, item):
    """ Print the decision_path of one sample
    
    Parameters
    ----------
    estimator : type of estimator is sklearn.tree.tree.DecisionTreeClassifier
    item : a data of training dataset

    Returns
    -------
    """
    sample_list = [item]
    node_indicator = estimator.decision_path(sample_list) # 返回实例所经历的路径的类 node_indicator
    # leave_id = estimator.apply(sample_list) # 返回实例所对应的节点ID
    node_index = node_indicator.indices[:]
    
    feature = estimator.tree_.feature # 所有节点的特征索引 -2为叶子节点
    threshold = estimator.tree_.threshold # 所有节点的阈值 -2为叶子节点
    value = estimator.tree_.value # 所有节点的N个类的个数 
    impurity = estimator.tree_.impurity # 所有节点的不纯度
    sample = estimator.tree_.n_node_samples # 所有节点所有的
    
    for node_id in node_index:
        if (item[feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        if threshold[node_id] == -2:
            print("node %s: leaf node, sample=%s, value=%s, gini=%.2f, label=%s" 
                  %(node_id, sample[node_id],
                 value[node_id],
                 impurity[node_id],
                 np.argmax(value[node_id], axis=1)))
            continue

        print("node %s: (X_test[i, %s](= %s) %s %.2f) sample=%s, value=%s, gini=%.2f, label=%s"
              % (node_id,
                 feature[node_id],
                 item[feature[node_id]],
                 threshold_sign,
                 threshold[node_id],
                 sample[node_id],
                 value[node_id],
                 impurity[node_id],
                 np.argmax(value[node_id], axis=1)))
        
def print_decision_path_ofsamplelist(estimator, itemlist):
    """ Print the decision_path of same sample list
    
    Parameters
    ----------
    estimator : type of estimator is sklearn.tree.tree.DecisionTreeClassifier
    itemlist : a data list of training dataset

    Returns
    -------
    """
    for id, item in enumerate(itemlist):
        print('Rules used to predict sample %s: ' % id)
        print_decision_path_ofsample(estimator, item)
        
def print_node_info(estimator, node_id):
    """ Print the node info of one estimator about node_id
    
    Parameters
    ----------
    estimator : type of estimator is sklearn.tree.tree.DecisionTreeClassifier
    node_id : a node id of tree nodes
    
    Returns
    -------
    """
    feature = estimator.tree_.feature # 所有节点的特征索引 -2为叶子节点
    threshold = estimator.tree_.threshold # 所有节点的阈值 -2为叶子节点
    value = estimator.tree_.value # 所有节点的N个类的个数 
    impurity = estimator.tree_.impurity # 所有节点的不纯度
    sample = estimator.tree_.n_node_samples # 所有节点所有的
    
    assert node_id >= 0
    assert node_id < estimator.tree_.node_count
    
    threshold_sign = "<="
    print("node %s: (feature[%s] %s %.2f) sample=%s, value=%s, gini=%.2f, label=%s"
              % (node_id,
                 feature[node_id],
                 threshold_sign,
                 threshold[node_id],
                 sample[node_id],
                 value[node_id],
                 impurity[node_id],
                 np.argmax(value[node_id], axis=1)))
    
    
def return_node_dataindex(estimator, leaf_node_id, data):
    """ Return the data index about leaf node id of one estimator
    
    Parameters
    ----------
    estimator : type of estimator is sklearn.tree.tree.DecisionTreeClassifier
    leaf_node_id : a leaf node id of tree nodes
    data : all train data
    
    Returns
    -------
        dataindex : array, shape = [len of data in leaf node]
    """
    nodelist = estimator.apply(data) # 返回实例所对应的节点ID
    dataindex = []
    for i, id in enumerate(nodelist):
        if id == leaf_node_id: dataindex.append(i)
    return dataindex