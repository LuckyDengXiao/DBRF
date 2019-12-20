# encoding=utf-8

import ModelUtils
import importlib
import numpy as np
from collections import Counter
importlib.reload(ModelUtils)

class Producer():

    def __init__(self, pass_train_x_list, pass_train_y_list, pass_test_x_list, pass_test_y_list):
        train_x = []
        for echo_data in pass_train_x_list:
            for data in echo_data:
                train_x.append(data)
        train_x = np.array(train_x)

        train_y = []
        for echo_data in pass_train_y_list:
            for data in echo_data:
                train_y.append(data)
        train_y = np.array(train_y)

        test_x = []
        for echo_data in pass_test_x_list:
            for data in echo_data:
                test_x.append(data)
        test_x = np.array(test_x)

        test_y = []
        for echo_data in pass_test_y_list:
            for data in echo_data:
                test_y.append(data)
        test_y = np.array(test_y)

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def prodect(self, num_class, feval, n_estimators=200):
        print("decision_tree")
        ModelUtils.decision_tree(self.train_x, self.train_y, self.test_x, self.test_y, feval=feval)
        print("random_forest")
        ModelUtils.random_forest(self.train_x, self.train_y, self.test_x, self.test_y, n_estimators=n_estimators, feval=feval)
        # print("gdbt_model")
        # ModelUtils.gdbt_model(self.train_x, self.train_y, self.test_x, self.test_y, n_estimators=n_estimators, feval=feval)
        # print("xgb_model")
        # ModelUtils.xgb_model(self.train_x, self.train_y, self.test_x, self.test_y, num_class=num_class, feval=feval)
        print("baseline_1_flod")
        ModelUtils.baseline_1_flod(self.train_x, self.train_y, self.test_x, self.test_y, num_class=num_class, feval=feval)
        print("baseline_model")
        ModelUtils.baseline_model(self.train_x, self.train_y, self.test_x, self.test_y, num_class=num_class, feval=feval)

