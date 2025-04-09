"""
实现训练器和优化器
"""

import numpy as np
from typing import List
import matplotlib.pyplot as plt
from .visualizer import Visualizer


class Trainer:
    def __init__(
        self,
        model,
        save_path_config,
        batch_size=128,
        epochs=10,
        learning_rate=0.01,
        lr_decay=0.99,
        l2_reg=0.001,
        random_seed=42,
    ):
        np.random.seed(random_seed)
        self.model = model
        self.save_path_config = save_path_config
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.l2_reg = l2_reg
        self.random_seed = random_seed
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.current_epoch = 0

    def train(self, train_data, train_labels, val_data, val_labels):
        num_train = train_data.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)

        best_val_acc = 0
        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # 每 10 个 epoch 可视化一次 Layer 0 的权重
            if (epoch + 1) % 10 == 0:
                Visualizer.visualize_weights(
                    self.model,
                    layer_idx=0,
                    epoch=epoch,
                    save_path=self.save_path_config.get_path(
                        f"weights_visualization_0_epoch_{epoch + 1}.png"
                    ),
                )

            # 训练
            for it in range(iterations_per_epoch):
                # 随机选择 batch_size 个样本作为每个批次的训练数据
                np.random.seed(self.random_seed + epoch + it)
                batch_mask = np.random.choice(num_train, self.batch_size)
                X_batch = train_data[batch_mask]
                y_batch = train_labels[batch_mask]

                loss = self._train_step(X_batch, y_batch)
                self.train_loss_history.append(loss)

            val_loss, val_acc = self.evaluate(val_data, val_labels)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)

            print(
                f"Epoch {epoch + 1}/{self.epochs}, train_loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.model.save_weights(
                    self.save_path_config.get_path("best_model.npy")
                )

            self.lr *= self.lr_decay

    def _train_step(self, X_batch, y_batch):
        # 前向传播
        scores = self.model.forward(X_batch)

        # 反向传播
        loss, dscores = self._compute_loss(scores, y_batch)
        self.model.backward(dscores)

        # 更新权重
        for layer in self.model.layers:
            if hasattr(layer, "weights"):
                layer.weights -= self.lr * (
                    layer.gradients["weights"] + self.l2_reg * layer.weights
                )
                layer.biases -= self.lr * layer.gradients["biases"]

        return loss

    def _compute_loss(self, scores, y):
        num_train = scores.shape[0]
        # 交叉熵损失
        loss = -np.sum(np.log(scores[range(num_train), y])) / num_train

        # L2 正则化
        reg_loss = 0
        for layer in self.model.layers:
            if hasattr(layer, "weights"):
                reg_loss += 0.5 * self.l2_reg * np.sum(layer.weights**2)
        loss += reg_loss

        # 计算交叉熵损失对模型输出的梯度
        dscores = scores.copy()
        dscores[range(num_train), y] -= 1
        dscores /= num_train

        return loss, dscores

    def evaluate(self, X, y):
        scores = self.model.forward(X)
        loss, _ = self._compute_loss(scores, y)
        y_pred = np.argmax(scores, axis=1)
        acc = np.mean(y_pred == y)
        return loss, acc
