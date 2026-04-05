# 这个脚本使用逻辑回归评估特征的线性可分性，输出分类报告。它将帮助我们了解特征在不同类别上的表现，为后续模型选择提供参考。

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report


X = np.load('features.npy')
y = np.load('labels.npy')

scaler = StandardScaler()
X = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=1000, multi_class='ovr')

y_pred = np.zeros_like(y)
for train_idx, test_idx in cv.split(X, y):
    clf.fit(X[train_idx], y[train_idx])
    y_pred[test_idx] = clf.predict(X[test_idx])

print(classification_report(y, y_pred, digits=4))

