#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_1nn_test.py
作用：
1. 读取语音 embedding 文件（features.npy）与标签（labels.npy）
2. 标准化特征向量
3. 用 1-近邻（KNeighborsClassifier, n_neighbors=1）评估特征区分度上限
4. 输出 3 折交叉验证的平均准确率，作为后续换模型/升维的参考基线
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier   # 可选，用于对比
from pathlib import Path

# ---------- 1. 载入数据 ----------
feat_file = Path(r"D:\专用轻量分类器\tiny_CNN32维\features_32_short_arcface_best.npy")
label_file = Path(r"D:\专用轻量分类器\tiny_CNN32维\labels_32_short_arcface_best.npy")

if not feat_file.exists() or not label_file.exists():
    raise FileNotFoundError("请把 features.npy 与 labels.npy 放到当前目录！")

X = np.load(feat_file)          # shape: (N, D)
y = np.load(label_file)         # shape: (N,)

print(f"数据加载完成：{X.shape[0]} 条样本，{X.shape[1]} 维特征，{len(np.unique(y))} 个类别")

# ---------- 2. 标准化 ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- 3. 1-NN 交叉验证 ----------
print("\n===== 1-Nearest-Neighbour (上限测试) =====")
acc_1nn = cross_val_score(KNeighborsClassifier(n_neighbors=1),
                          X_scaled, y,
                          cv=3,   # 3 折即可，数据量大可改成 5
                          scoring='accuracy',
                          n_jobs=-1).mean()
print(f"1-NN 3 折交叉验证平均准确率：{acc_1nn:.4f}")

# ---------- 4. （可选）RandomForest 对比 ----------
print("\n===== RandomForest-100 （对比） =====")
acc_rf = cross_val_score(RandomForestClassifier(n_estimators=100,
                                                random_state=42),
                         X_scaled, y,
                         cv=3,
                         scoring='accuracy',
                         n_jobs=-1).mean()
print(f"RandomForest-100 3 折交叉验证平均准确率：{acc_rf:.4f}")

# ---------- 5. 小结 ----------
print(f"\n=== 结果摘要 ===")
print(f"1-NN 上限 : {acc_1nn:.1%}")
print(f"RF-100    : {acc_rf:.1%}")
print("若 1-NN > 80% → 特征区分度足够，换非线性模型即可；"
      "若 < 50% → 需升维或重新提 embedding。")