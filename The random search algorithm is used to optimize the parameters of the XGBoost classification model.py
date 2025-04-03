# -*- coding: utf-8 -*-
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef

# 加载数据集
wine = load_wine()
X = wine.data
y = wine.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=24)

print("---------------------使用默认参数----------------------------")
# 初始化XGBoost分类器
model_default = XGBClassifier(random_state=24)
# 训练
model_default.fit(X_train, y_train)
# 预测
y_pred_default = model_default.predict(X_test)

# 输出默认参数下的评估指标
acc_default = accuracy_score(y_test, y_pred_default)
print("默认参数 accuracy:", acc_default)

precision_default = precision_score(y_test, y_pred_default, average='weighted')
recall_default = recall_score(y_test, y_pred_default, average='weighted')
f1_default = f1_score(y_test, y_pred_default, average='weighted')
auc_default = roc_auc_score(y_test, model_default.predict_proba(X_test), multi_class='ovr')
mcc_default = matthews_corrcoef(y_test, y_pred_default)
conf_mat_default = confusion_matrix(y_test, y_pred_default)

print("精确率:", precision_default)
print("召回率:", recall_default)
print("F1分数:", f1_default)
print("AUC：", auc_default)
print("MCC：", mcc_default)
print("混淆矩阵:\n", conf_mat_default)

print("---------------------参数寻优----------------------------")
t1 = time.time()

# 定义参数分布
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 2, 3, 4, 5],
}

# 初始化XGBoost分类器
model = XGBClassifier(random_state=24)

# 初始化随机搜索对象
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100,
                                   cv=5, scoring='accuracy', random_state=24, verbose=2)

# 执行随机搜索
random_search.fit(X_train, y_train)
t2 = time.time()

# 输出最优参数
print("Best parameters:")
print(random_search.best_params_)
print("time:", t2-t1)

print("---------------------最优模型----------------------------")
# 使用最优参数创建最优模型
best_params = random_search.best_params_
model_best = random_search.best_estimator_

# 训练
model_best.fit(X_train, y_train)
# 预测
y_pred_best = model_best.predict(X_test)

# 输出最优模型下的评估指标
acc_best = accuracy_score(y_test, y_pred_best)
print("最优参数 accuracy:", acc_best)

precision_best = precision_score(y_test, y_pred_best, average='weighted')
recall_best = recall_score(y_test, y_pred_best, average='weighted')
f1_best = f1_score(y_test, y_pred_best, average='weighted')
auc_best = roc_auc_score(y_test, model_best.predict_proba(X_test), multi_class='ovr')
mcc_best = matthews_corrcoef(y_test, y_pred_best)
conf_mat_best = confusion_matrix(y_test, y_pred_best)

print("精确率:", precision_best)
print("召回率:", recall_best)
print("F1分数:", f1_best)
print("AUC：", auc_best)
print("MCC：", mcc_best)
print("混淆矩阵:\n", conf_mat_best)

