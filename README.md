# EcoForest（DBRF）
Eco-Forest is more effective version of gcForest

# GOAL
- unbalanced dataset
  - Ref. imbalanced-learn of sklearn. http://contrib.scikit-learn.org/imbalanced-learn/stable/
- missing value
  - Ref. 
- feature combination
  - Ref. FM, FFM
- layer/deep structure
  - Ref. gcForest, eForest
- discrete or continuous variable
- abnormal data

# To Do
1. Analysis sklearn Tree source code (DONE)
2. Analysis sklearn ensemble source code (DONE)
3. Add Utils code: ForestUtils.py (DONE)
4. Add Utils code: EnhancedDTree.py (DONE)
5. Add Utils code: EnhancedForest.py (DONE)
6. Case Study: Forest Driver dataset test (DONE)
7. LayerForest v0.1 - LayerDTree (DONE)
  - finish layer structure of DTree
  - spliting the data by the gini value of leaf node
  - test model by globel vaild data 
  - output the test data
8. LayerForest v0.2 - LayerDTree (DONE)
  - driver dataset
  - EnhancedDTree.py
9. LayerForest v0.3 - LayerDTree (DONE)
  - uci adult dataset
  - compare with xgb, rf, decisiontree
  - EnhancedDTree.py
10. LayerForest v0.4 - LayerDTree++ (DONE)
  - eliminate the overfit result
  - k-fold
11. LayerForest v0.5 - LayerForest (DONE)
  - EnhancedForest.py
  - bug: misconvergence
11. LayerForest v0.6 - LayerForest (ING)
  - [D] debug: eliminate miscovergence [Dropout \ Batch Normalization]
  - debug: eliminate overfit
  - debug: eliminate overquick covergence [?]
  - exceed xgb, rf, decisiontree
  - all to do:
    - k-fold train [v]
    - avg predict [v]
    - threshold of all imp - avg [v]
    - dropout by score of est [v]
    - dropout by score of tree [x]
    - LR predict [v]
    - vaild data split [v]
12. LayerForest v0.7 (Done)
  - Simplify Procedure
    - Data Load Utils
  - Multiclass Support
13. LayerForest v0.8 (ING)
  - Simplify Procedure
    - Model Utils
    - DecomposerForest
    - AlgorithmUtils
14. LayerForest v1.0 - ecoForest
  - Vaild Data Split
  - MaxLayer Control
  - Train/Vaild Loss Guide
  - Freq/Lift/Support Score
  - K-Flod
  - Smart Early Stop
  - LR Stacker

# Other Files
EnhancedForest_multiclass_v0.2: before AlgorithmUtils. 12.10

# Using Data Introduction
- Kaggle Datasets:
  - Driver: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
  - Credit Card: https://www.kaggle.com/mlg-ulb/creditcardfraud
- UCI Datasets
  - Letter: https://archive.ics.uci.edu/ml/datasets/letter+recognition
  - Adult: https://archive.ics.uci.edu/ml/datasets/adult
  - Yeast: https://archive.ics.uci.edu/ml/datasets/Yeast
  - Skin: https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
  - Poker: https://archive.ics.uci.edu/ml/datasets/Poker+Hand


Happy Hacking.
