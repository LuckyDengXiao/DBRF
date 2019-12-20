# encoding=utf-8

import datetime
import time
import numpy as np

class LogRecord():

    def __init__(self):
        tm = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S.txt')
        self.log_path = "./logs/" + tm

    def log(self, log_str, prefix=""):
        with open(self.log_path, "a") as f:
            tm = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
            f.write(tm+"-"+prefix+":"+log_str+"\n")

    def log_array(self, array, prefix=""):
        log_str = np.array_str(array)
        self.log(log_str, prefix)

    def log_value(self, value, prefix=""):
        log_str = str(value)
        self.log(log_str, prefix)

    def log_all_array(self, array, prefix=""):
        log_str = ",".join([str(i) for i in array])
        self.log(log_str, prefix)

class  ResultRecord(LogRecord):

    def __init__(self, flag):
        tm = datetime.datetime.fromtimestamp(time.time()).strftime(flag+'%Y_%m_%d_%H_%M_%S.txt')
        self.log_path = "./results/" + tm
        