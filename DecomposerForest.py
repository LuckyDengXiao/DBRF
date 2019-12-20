# encoding=utf-8

import EnhancedForest_multiclass
import EnhancedForest
import EnhancedForest_producer
import LogUtils
from sklearn import metrics
import numpy as np
import datetime
import time

class STATUS_FIT():
    STATUS_BREAK = "BREAK"
    STATUS_CONTINUE = "CONTINUE"
    STATUS_NOTPASS = "NOTPASS"
    MAX_LAYER = 100

class DecomposerForest():
    
    def __init__(self, X_train, y_train, X_sub, y_sub, num_class, flag, isLRStacker=True):
        self._statistical_info()
        self._first_layer_init(X_train, y_train, X_sub, y_sub, isLRStacker)
        self.num_class = num_class
        self.loger = LogUtils.ResultRecord(flag)
        self.loger.log("isLRStacker:"+str(isLRStacker), "init")
        self.set_parameter()

    def _statistical_info(self):
        # 统计信息
        self.train_loss_lt = []
        self.pass_train_loss_lt = []
        self.pass_train_loss_lt_now = []
        self.vaild_loss_lt = []
        self.pass_vaild_loss_lt = []
        self.pass_vaild_loss_lt_now = []
        self.test_loss_lt = []
        self.pass_test_loss_lt = []
        self.pass_test_loss_lt_now = []
        self.pass_data_rate_lt = []

    def _first_layer_init(self, X_train, y_train, X_sub, y_sub, isLRStacker):
        # 数据
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        # X = X_train.copy()
        # y = y_train.copy()

        self.X_test = X_sub.copy()
        self.real_y = y_sub.copy()
        self.test_y = np.array(([0.0] * len(X_sub)))
        self.all_data_mask = np.array([False] * len(X_sub))
        # data_mask = np.array([False] * len(X_sub))

        # 不均衡数据进行layer
        # X_train_np = X
        # y_train_np = y
        # maxlayer = 100

        # 不降低不更新
        self.last_train_loss = 0
        self.last_vaild_loss = 0

        # self.counter = 0
        self.early_stop = 0
        self.last_is_early_stop = False
        # self.early_stop_up = 0

        self.enhancedDTree = EnhancedForest_multiclass.EnhancedForest(len(X_train), len(X_sub), isLRStacker=isLRStacker)
        # self.enhancedDTree = EnhancedForest.EnhancedForest(len(X_train), len(X_sub), isLRStacker=isLRStacker)

    def _get_loss(self):
        train_loss = self.enhancedDTree.getTrainLoss(self.feval)
        pass_train_loss = self.enhancedDTree.getPassTrainLoss(self.feval)
        pass_train_loss_now = self.enhancedDTree.getPassTrainLossNow(self.feval)
        vaild_loss = self.enhancedDTree.getVaildLoss(self.feval)
        pass_vaild_loss = self.enhancedDTree.getPassVaildLoss(self.feval)
        pass_vaild_loss_now = self.enhancedDTree.getPassVaildLossNow(self.feval)
        test_loss = self.enhancedDTree.getTestLoss(self.feval)
        pass_test_loss = self.enhancedDTree.getPassTestLoss(self.feval)
        pass_test_loss_now = self.enhancedDTree.getPassTestLossNow(self.feval)

        print("train loss", train_loss)
        print("pass train loss", pass_train_loss)
        print("pass train loss now", pass_train_loss_now)
        print("vaild loss", vaild_loss)
        print("pass vaild loss", pass_vaild_loss)
        print("pass vaild loss now", pass_vaild_loss_now)
        print("test loss", test_loss)
        print("pass test loss", pass_test_loss)
        print("pass test loss now", pass_test_loss_now)

        return train_loss, pass_train_loss, pass_train_loss_now, \
            vaild_loss, pass_vaild_loss, pass_vaild_loss_now, \
            test_loss, pass_test_loss, pass_test_loss_now

    def _record_loss(self, layer_loss, pass_data_rate):
        train_loss, pass_train_loss, pass_train_loss_now, \
            vaild_loss, pass_vaild_loss, pass_vaild_loss_now, \
            test_loss, pass_test_loss, pass_test_loss_now = layer_loss
        self.train_loss_lt.append(train_loss[1])
        self.pass_train_loss_lt.append(pass_train_loss[1])
        self.pass_train_loss_lt_now.append(pass_train_loss_now[1])
        self.vaild_loss_lt.append(vaild_loss[1])
        self.pass_vaild_loss_lt.append(pass_vaild_loss[1])
        self.pass_vaild_loss_lt_now.append(pass_vaild_loss_now[1])
        self.test_loss_lt.append(test_loss[1])
        self.pass_test_loss_lt.append(pass_test_loss[1])
        self.pass_test_loss_lt_now.append(pass_test_loss_now[1])

        if len(self.pass_data_rate_lt) == 0:
            self.pass_data_rate_lt.append(pass_data_rate)
        else:
            self.pass_data_rate_lt.append(self.pass_data_rate_lt[-1]+pass_data_rate)

    def _pass_update_info(self, layer, layer_loss, data_mask, all_false_data_index):
        ''' update these infos
        self.last_train_loss, self.last_vaild_loss
        self.last_is_early_stop, self.early_stop
        self.all_data_mask
        '''
        self.last_train_loss = layer_loss[0][1]
        self.last_vaild_loss = layer_loss[3][1]
        
        # X_train_np = self.enhancedDTree.X_train_np
        # y_train_np = self.enhancedDTree.y_train_np
        
        # 打印信息
        pass_data_id = data_mask[data_mask==True]
    #     all_false_data_index = np.where(all_data_mask == False)[0]
        X_test_np = self.X_test[all_false_data_index]
        print("%d [p:%d/np:%d] " % (layer, len(pass_data_id), len(X_test_np) - len(pass_data_id)))#, end="")

        if len(pass_data_id) == 0: 
            if not self.last_is_early_stop: self.early_stop = 0
            self.early_stop += 1
            self.last_is_early_stop = True
        else:
            self.last_is_early_stop = False
        
    #     tmp_all_data_mask = all_false_data_index[~data_mask]
    #     test_y[tmp_all_data_mask] = p_test[~data_mask]
        # tmp_test_loss = feval(y_sub, test_y)
        # best_test_y = test_y.copy()
        # print("best test loss:", tmp_test_loss)
        
        # _record_loss
        pass_data_rate = len(data_mask[data_mask==True])/len(self.X_test)
        self._record_loss(layer_loss, pass_data_rate)
        # 打印信息结束

        self.all_data_mask[~self.all_data_mask] = data_mask
            

    def _fit_layer(self, X, y, isFirst, layer):
        #从fit函数调用过来
        clf, data_mask, all_false_data_index, p_test = \
            self.enhancedDTree.TrainModelLayer(X, y, self.X_test, self.all_data_mask, self.test_y, self.real_y, verbose=False, \
                                          feval=self.feval, dropout=self.dropout, criterion=self.criterion, random_state=1022+layer, \
                                            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,\
                                          isFirst=isFirst, num_class=self.num_class, kfold=self.kfold, \
                                          n_estimators=self.n_estimators\
                                         )
        #输出Loss信息
        layer_loss = self._get_loss()
        trainloss = layer_loss[0][1]
        vaildloss = layer_loss[3][1]
        trainpassnow = layer_loss[2][1]
        vaildpassnow = layer_loss[5][1]

        if trainloss <= self.last_train_loss and vaildloss < self.last_vaild_loss:
            #这说明效果没有变好，不更新
    #     if vaild_loss[1] < last_vaild_loss: 
            if not isFirst: self.enhancedDTree.remove_last_items()
    #         early_stop_up += 1
            # if layer > STATUS_FIT.MAX_LAYER or self.early_stop > 5 or early_stop_up > 15:
            if layer > STATUS_FIT.MAX_LAYER or self.early_stop > 20:
                # break
                return STATUS_FIT.STATUS_BREAK
            # continue
            return STATUS_FIT.STATUS_NOTPASS
        # v0.8 add new condition
        if trainpassnow < trainloss and vaildpassnow < vaildloss:
            if not isFirst: self.enhancedDTree.remove_last_items()
            if layer > STATUS_FIT.MAX_LAYER or self.early_stop > 20:
                return STATUS_FIT.STATUS_BREAK
            return STATUS_FIT.STATUS_NOTPASS

        self._pass_update_info(layer, layer_loss, data_mask, all_false_data_index)
            
    #     if X_train_np.shape[0] < 10 or layer > maxlayer or y_train_np[y_train_np==1].shape[0] <= 10 or early_stop > 5:
        if layer > STATUS_FIT.MAX_LAYER or self.early_stop > 20:
            # break
            return STATUS_FIT.STATUS_BREAK
        return STATUS_FIT.STATUS_CONTINUE

    def set_parameter(self, dropout=None, criterion='gini', max_depth=None, min_samples_leaf=1):
        self.dropout = dropout
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.loger.log("dropout:"+str(dropout)+",criterion:"+str(criterion)\
            +",max_depth:"+str(max_depth)+",min_samples_leaf:"+str(min_samples_leaf)\
              , "set_parameter")

    def log_result(self):
        self.loger.log_all_array(self.train_loss_lt, "train")
        self.loger.log_all_array(self.pass_train_loss_lt, "pass")
        self.loger.log_all_array(self.pass_train_loss_lt_now, "now")
        self.loger.log_all_array(self.vaild_loss_lt, "vaild")
        self.loger.log_all_array(self.pass_vaild_loss_lt, "pass")
        self.loger.log_all_array(self.pass_vaild_loss_lt_now, "all")
        self.loger.log_all_array(self.test_loss_lt, "test")
        self.loger.log_all_array(self.pass_test_loss_lt, "pass")
        self.loger.log_all_array(self.pass_test_loss_lt_now, "all")
        self.loger.log_all_array(self.pass_data_rate_lt, "data")
        self.loger.log_value(np.max(np.array(self.test_loss_lt)), "npmax")
        self.loger.log_all_array(list(enumerate(self.test_loss_lt)), "testloss")

    def fit(self, n_estimators=200, kfold=5, feval=metrics.accuracy_score):
        self.kfold = kfold
        self.feval = feval
        self.n_estimators = n_estimators
        self.loger.log("kfold:"+str(kfold)+",feval:"+str(feval)+",n_estimators:"+str(n_estimators), "fit")
        X_train_np = self.X_train
        y_train_np = self.y_train 
        layer = 0
        while 1:
            layer += 1
            print()
            tm = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            print(tm, "layer:", layer)
            isFirst = True if layer == 1 else False

            #更新本层使用的训练集和标签
            X = X_train_np
            y = y_train_np

            #调用DBRF进行训练
            status_fit = self._fit_layer(X, y, isFirst, layer)

            #训练完毕，得到返回的状态
            if status_fit == STATUS_FIT.STATUS_BREAK:
                print("STATUS_BREAK")
                break
            elif status_fit == STATUS_FIT.STATUS_CONTINUE:
                X_train_np = self.enhancedDTree.X_train_np
                y_train_np = self.enhancedDTree.y_train_np

                if isFirst: continue
                pass_train_x_list = self.enhancedDTree.pass_data_x_list
                pass_train_y_list = self.enhancedDTree.pass_data_y_list
                pass_test_x_list = self.enhancedDTree.pass_data_test_list
                pass_test_y_list = self.enhancedDTree.pass_real_test_list
 
                # producerForest = EnhancedForest_producer.Producer(pass_train_x_list, pass_train_y_list, pass_test_x_list, pass_test_y_list)
                # producerForest.prodect(10, self.feval)

            elif status_fit == STATUS_FIT.STATUS_NOTPASS:
                print("not pass")
        self.log_result()