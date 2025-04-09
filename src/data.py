"""
实现数据加载和预处理
"""

import os
import numpy as np
from typing import Tuple


def load_cifar10():
    # 设置随机数种子
    np.random.seed(42)

    def unpickle(file):
        import pickle

        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict

    X_train = []
    y_train = []

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for i in range(1, 6):
        data_path = os.path.join(script_dir, f"data/data_batch_{i}")
        batch = unpickle(data_path)
        X_train.append(batch[b"data"])
        y_train.extend(batch[b"labels"])

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.array(y_train)

    test_path = os.path.join(script_dir, "data/test_batch")
    test_batch = unpickle(test_path)
    X_test = test_batch[b"data"]
    y_test = np.array(test_batch[b"labels"])

    # 归一化
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # 划分验证集（使用固定的随机排列）
    val_size = 5000
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test
