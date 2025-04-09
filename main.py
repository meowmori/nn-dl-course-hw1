import numpy as np
from src.model import NeurNet
from src.trainer import Trainer
from src.data import load_cifar10
from src.visualizer import Visualizer
from src.utils import SavePathConfig


def main():
    np.random.seed(42)

    # 实验结果保存路径
    save_path_config = SavePathConfig()

    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()

    # 创建模型
    input_dim = 3072  # 32x32x3
    hidden_dim = 512
    num_classes = 10
    model = NeurNet(
        input_dim, hidden_dim, num_classes, activation_class="relu"
    )  # activation_class 可选 'relu', 'sigmoid', 'tanh'

    layer_idx = 0
    for layer in model.layers:
        if hasattr(layer, "weights"):
            print(f"layer {layer_idx} - linear - weights shape:", layer.weights.shape)
        else:
            print(f"layer {layer_idx} - activation - type:", type(layer).__name__)
        layer_idx += 1

    # 设置训练超参数
    batch_size = 128
    epochs = 100
    learning_rate = 0.03
    lr_decay = 0.98  # 学习率衰减系数
    l2_reg = 0.0001  # L2正则化系数

    # 模型训练
    trainer = Trainer(
        model,
        save_path_config,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        l2_reg=l2_reg,
        random_seed=42,
    )
    trainer.train(X_train, y_train, X_val, y_val)

    # 训练结果可视化
    Visualizer.plot_training_history(
        trainer, save_path=save_path_config.get_path("training_history.png")
    )
    Visualizer.visualize_weights(
        model,
        layer_idx=0,
        epoch=trainer.current_epoch,
        save_path=save_path_config.get_path(f"weights_visualization_0_epoch_final.png"),
    )

    # 模型评估
    model.load_weights(save_path_config.get_path("best_model.npy"))
    test_loss, test_acc = trainer.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # 保存实验结果
    with open(save_path_config.get_path("results.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")

    print(f"Results saved to: {save_path_config.exp_dir}")


if __name__ == "__main__":
    main()
