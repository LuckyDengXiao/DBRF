import os
import numpy as np
import pandas as pd
import time
import os.path as osp
import DataHelper
import importlib
from collections import Counter
importlib.reload(DataHelper)

X_train,y_train, X_sub, y_sub = DataHelper.get_yeast_data()

Counter(y_train)
Counter(y_sub)

import ModelUtils
import importlib
from collections import Counter
importlib.reload(ModelUtils)
np.set_printoptions(threshold=np.inf)

from sklearn import metrics
def acc_metrix_mult(a, p):
    return "acc", metrics.accuracy_score(a, p)

import EnhancedForest_multiclass
import EnhancedForest_producer
import DecomposerForest
import LogUtils
import AlgorithmUtils
import importlib
importlib.reload(DecomposerForest)
importlib.reload(LogUtils)
importlib.reload(EnhancedForest_multiclass)
importlib.reload(EnhancedForest_producer)
importlib.reload(AlgorithmUtils)
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings('ignore')

decoForest = DecomposerForest.DecomposerForest(X_train, y_train, X_sub, y_sub, num_class=10, flag="yeast", isLRStacker=True)
decoForest.set_parameter(criterion="entropy", dropout=0.8, min_samples_leaf=1)
#调用模型，开始训练
decoForest.fit(n_estimators=200, kfold=2, feval=acc_metrix_mult)