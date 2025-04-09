"""
实现可视化工具
"""

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    @staticmethod
    def plot_training_history(trainer, save_path):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # 训练loss曲线
        ax1.plot(trainer.train_loss_history)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Training Loss")
        ax1.set_title("Training Loss vs Iteration")

        # 验证loss曲线
        ax2.plot(trainer.val_loss_history)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Loss")
        ax2.set_title("Validation Loss vs Epoch")

        # 验证准确率曲线
        ax3.plot(trainer.val_acc_history)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Validation Accuracy")
        ax3.set_title("Validation Accuracy vs Epoch")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def visualize_weights(model, layer_idx=0, epoch=None, save_path=None):
        weights = model.layers[layer_idx].weights

        # 可视化前 64 个神经元的权重
        n_neurons = min(64, weights.shape[1])
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        fig.suptitle(
            f'Layer {layer_idx} Weights (Epoch {epoch if epoch is not None else "final"})'
        )

        for i, ax in enumerate(axes.flat):
            if i < n_neurons:
                img = weights[:, i].reshape(3, 32, 32).transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())
                ax.imshow(img)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            save_path
            or f"weights_visualization_{layer_idx}_epoch_{epoch if epoch is not None else "final"}.png"
        )
        plt.close()
