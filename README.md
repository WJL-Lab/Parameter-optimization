# The random search algorithm is used to optimize the parameters of the XGBoost classification model

The results of the run:

```
---------------------使用默认参数----------------------------
默认参数 accuracy: 0.9555555555555556
精确率: 0.9586868686868687
召回率: 0.9555555555555556
F1分数: 0.9548880748880749
AUC：0.9992816091954023
MCC：0.9334862385321101
混淆矩阵:
 [[19  0  0]
 [ 1 14  1]
 [ 0  0 10]]
---------------------参数寻优----------------------------
Fitting 5 folds for each of 100 candidates, totalling 500 fits
[CV] END colsample_bytree=0.8, learning_rate=0.1, max_depth=8, min_child_weight=4, n_estimators=200, subsample=0.7; total time=   0.1s
[CV] END colsample_bytree=0.8, learning_rate=0.1, max_depth=8, min_child_weight=4, n_estimators=200, subsample=0.7; total time=   0.1s
[CV] END colsample_bytree=0.8, learning_rate=0.1, max_depth=8, min_child_weight=4, n_estimators=200, subsample=0.7; total time=   0.1s
[CV] END colsample_bytree=0.8, learning_rate=0.1, max_depth=8, min_child_weight=4, n_estimators=200, subsample=0.7; total time=   0.1s
......
[CV] END colsample_bytree=0.7, learning_rate=0.05, max_depth=3, min_child_weight=3, n_estimators=300, subsample=0.7; total time=   0.2s
[CV] END colsample_bytree=0.7, learning_rate=0.05, max_depth=3, min_child_weight=3, n_estimators=300, subsample=0.7; total time=   0.3s
[CV] END colsample_bytree=0.7, learning_rate=0.05, max_depth=3, min_child_weight=3, n_estimators=300, subsample=0.7; total time=   0.2s
Best parameters:
​
{'subsample': 1.0, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 10, 'learning_rate': 0.01, 'colsample_bytree': 0.6}
time: 173.85961079597473
---------------------最优模型----------------------------
最优参数 accuracy: 0.9777777777777777
精确率: 0.9797979797979799
召回率: 0.9777777777777777
F1分数: 0.9779484553678103
AUC：1.0
MCC：0.9664959643957367
混淆矩阵:
 [[19  0  0]
 [ 0 15  1]
 [ 0  0 10]]
```



![](D:\Jupyter Notebook\00-xin mei ti\利用随机搜索算法对XGBoost分类模型参数寻优\result.png)
