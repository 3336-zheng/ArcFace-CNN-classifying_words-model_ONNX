# view_features.py
#为了查看特征和标签文件的内容，了解特征矩阵的形状、标签数量、特征范围、均值、标准差以及标签分布情况。这有助于我们更好地理解数据的结构和分布，为后续的数据处理和模型训练提供参考。

import numpy as np
import json
import pandas as pd


def view_features_and_labels():
    """查看特征和标签文件内容"""

    # 加载特征数据
    features = np.load("features.npy")
    labels = np.load("labels.npy")

    # 加载指令映射
    with open(r"D:\专用轻量分类器\command_mapping_second_only_one.json", 'r', encoding='utf-8') as f:
        command_mapping = json.load(f)

    reverse_mapping = {v: k for k, v in command_mapping.items()}

    print("=" * 50)
    print("特征数据概览")
    print("=" * 50)
    print(f"特征矩阵形状: {features.shape}")
    print(f"标签数量: {len(labels)}")

    print(f"\n前60个样本的特征:")
    for i in range(min(60, len(features))):
        label_name = reverse_mapping.get(labels[i], "未知")
        print(f"样本 {i + 1} (标签: {label_name}):")
        print(f"  特征向量: {features[i]}")
        print(f"  特征范围: [{np.min(features[i]):.4f}, {np.max(features[i]):.4f}]")
        print()

    print(f"\n特征统计:")
    print(f"  所有特征范围: [{np.min(features):.4f}, {np.max(features):.4f}]")
    print(f"  特征均值: {np.mean(features):.4f}")
    print(f"  特征标准差: {np.std(features):.4f}")

    print(f"\n标签分布:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        label_name = reverse_mapping.get(label, "未知")
        print(f"  {label_name}: {count} 个样本")


if __name__ == "__main__":
    view_features_and_labels()