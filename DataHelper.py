# encoding=utf-8

import os
import numpy as np
import pandas as pd
import time
import os.path as osp
import re
from sklearn.model_selection import train_test_split

class FeatureParser(object):
    def __init__(self, desc):
        desc = desc.strip()
        if desc == "C":
            self.f_type = "number"
        else:
            self.f_type = "categorical"
            f_names = [d.strip() for d in desc.split(",")]
            # missing value
            f_names.insert(0, "?")
            self.name2id = dict(zip(f_names, range(len(f_names))))

    def get_float(self, f_data):
        f_data = f_data.strip()
        if self.f_type == "number":
            return float(f_data)
        return float(self.name2id[f_data])

    def get_data(self, f_data):
        f_data = f_data.strip()
        if self.f_type == "number":
            return float(f_data)
        data = np.zeros(len(self.name2id), dtype=np.float32)
        data[self.name2id[f_data]] = 1
        return data

    def get_fdim(self):
        """
        get feature dimension
        """
        if self.f_type == "number":
            return 1
        return len(self.name2id)

def get_adult_data():
    train_data_path = osp.join(".\\datasetes", "adult", "adult.data")
    test_data_path = osp.join(".\\datasetes", "adult", "adult.test")
    feature_desc_path = osp.join(".\\datasetes", "adult", "features")

    # get feature parsers
    f_parsers = []
    with open(feature_desc_path) as f:
        for row in f.readlines():
            f_parsers.append(FeatureParser(row))

    # get X_train y_train
    with open(train_data_path) as f:
        rows = [row.strip().split(",") for row in f.readlines() if len(row.strip()) > 0 and not row.startswith("|")]
    n_datas = len(rows)

    cate_as_onehot = 0
    if cate_as_onehot:
        X_dim = np.sum([f_parser.get_fdim() for f_parser in f_parsers])
        X = np.zeros((n_datas, X_dim), dtype=np.float32)
    else:
        X = np.zeros((n_datas, 14), dtype=np.float32)
    y = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        assert len(row) == 15, "len(row) wrong, i={}".format(i)
        foffset = 0
        for j in range(14):
            if cate_as_onehot:
                fdim = f_parsers[j].get_fdim()
                X[i, foffset:foffset+fdim] = f_parsers[j].get_data(row[j].strip())
                foffset += fdim
            else:
                X[i, j] = f_parsers[j].get_float(row[j].strip())
        y[i] = 0 if row[-1].strip().startswith("<=50K") else 1
    print("X.shape:", X.shape, "y.shape:", y.shape)
    X_train = X
    y_train = y

    # get X_sub y_sub
    with open(test_data_path) as f:
        rows = [row.strip().split(",") for row in f.readlines() if len(row.strip()) > 0 and not row.startswith("|")]
    n_datas = len(rows)

    cate_as_onehot = 0
    if cate_as_onehot:
        X_dim = np.sum([f_parser.get_fdim() for f_parser in f_parsers])
        X = np.zeros((n_datas, X_dim), dtype=np.float32)
    else:
        X = np.zeros((n_datas, 14), dtype=np.float32)
    y = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        assert len(row) == 15, "len(row) wrong, i={}".format(i)
        foffset = 0
        for j in range(14):
            if cate_as_onehot:
                fdim = f_parsers[j].get_fdim()
                X[i, foffset:foffset+fdim] = f_parsers[j].get_data(row[j].strip())
                foffset += fdim
            else:
                X[i, j] = f_parsers[j].get_float(row[j].strip())
        y[i] = 0 if row[-1].strip().startswith("<=50K") else 1
    print("X.shape:", X.shape, "y.shape:", y.shape)
    X_sub = X
    y_sub = y

    return X_train, y_train, X_sub, y_sub


def get_letter_data():
    data_path = osp.join(".\\datasetes", "letter", "letter-recognition.data")
    with open(data_path) as f:
        rows = [row.strip().split(',') for row in f.readlines()]
    n_datas = len(rows)
    X = np.zeros((n_datas, 16), dtype=np.float32)
    y = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        X[i, :] = list(map(float, row[1:]))
        y[i] = ord(row[0]) - ord('A')
    # train_idx, test_idx = train_test_split(range(n_datas), random_state=0, train_size=0.8, stratify=y)
    X_train, y_train = X[:16000], y[:16000]
    X_test, y_test = X[16000:], y[16000:]
    # X_train, y_train = X[train_idx], y[train_idx]
    # X_test, y_test = X[test_idx], y[test_idx]
    print("X.shape:", X_train.shape, "y.shape:", y_train.shape)
    print("X.shape:", X_test.shape, "y.shape:", y_test.shape)
    return X_train, y_train, X_test, y_test


def get_yeast_data():
    id2label = {}
    label2id = {}
    label_path = osp.abspath( osp.join(".\\datasetes", "yeast", "yeast.label") )
    with open(label_path) as f:
        for row in f:
            cols = row.strip().split(" ")
            id2label[int(cols[0])] = cols[1]
            label2id[cols[1]] = int(cols[0])

    data_path = osp.abspath( osp.join(".\\datasetes", "yeast", "yeast.data") )
    with open(data_path) as f:
        rows = f.readlines()
    n_datas = len(rows)
    X = np.zeros((n_datas, 8), dtype=np.float32)
    y = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        cols = re.split(" +", row.strip())
        #print(list(map(float, cols[1:1+8])))
        X[i,:] = list(map(float, cols[1:1+8]))
        y[i] = label2id[cols[-1]]
    train_idx, test_idx = train_test_split(range(n_datas), random_state=0, train_size=0.7, stratify=y)
    print("X.shape:", X[train_idx].shape, "y.shape:", y[train_idx].shape)
    print("X.shape:", X[test_idx].shape, "y.shape:", y[test_idx].shape)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def get_mnist_data():
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if self.data_set == 'train':
        X = X_train
        y = y_train
    elif self.data_set == 'train-small':
        X = X_train[:2000]
        y = y_train[:2000]
    elif self.data_set == 'test':
        X = X_test
        y = y_test
    elif self.data_set == 'test-small':
        X = X_test[:1000]
        y = y_test[:1000]
    elif self.data_set == 'all':
        X = np.vstack((X_train, X_test))
        y = np.vstack((y_train, y_test))
    else:
        raise ValueError('MNIST Unsupported data_set: ', self.data_set)

    # normalization
    if self.norm:
        X = X.astype(np.float32) / 255
    X = X[:,np.newaxis,:,:]
    X = self.init_layout_X(X)
    y = self.init_layout_y(y)
    self.X = X
    self.y = y
    
def get_poker_data():
    #now for train
    #data_path = osp.join(".\\datasetes", "poker", "poker-hand-testing.data")
    data_path = osp.join(".\\datasetes", "poker", "poker-hand-training-true.data")
    with open(data_path) as f:
        rows = [row.strip().split(',') for row in f.readlines()]
    n_datas = len(rows)
    X_train = np.zeros((n_datas, 10), dtype=np.int32)
    y_train = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        X_train[i, :] = list(map(int, row[0:0+10]))
        y_train[i] = int(row[-1])
        
    #now for test
    #data_path = osp.join(".\\datasetes", "poker", "poker-hand-training-true.data")
    data_path = osp.join(".\\datasetes", "poker", "poker-hand-testing.data")
    with open(data_path) as f:
        rows = [row.strip().split(',') for row in f.readlines()]
    n_datas = len(rows)
    X_test = np.zeros((n_datas, 10), dtype=np.int32)
    y_test = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        X_test[i, :] = list(map(int, row[0:0+10]))
        y_test[i] = int(row[-1])
        
    X_test_short = X_test[:100000]
    y_test_short = y_test[:100000]
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test_short.shape, "y_test:", y_test_short.shape)
    return X_train, y_train, X_test_short, y_test_short