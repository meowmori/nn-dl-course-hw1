"""
实现三层神经网络
"""

import numpy as np
import pickle
from .layers import Linear, ReLU, Sigmoid, Tanh


class NeurNet:
    def __init__(self, input_dim, hidden_dim, num_classes, activation_class="relu"):
        # 指定激活函数
        self.activation_dict = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}
        activation = self.activation_dict.get(activation_class.lower(), ReLU)

        self.layers = [
            Linear(input_dim, hidden_dim),
            activation(),
            Linear(hidden_dim, hidden_dim),
            activation(),
            Linear(hidden_dim, num_classes),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return self.softmax(x)

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def save_weights(self, path):
        weights = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                weights.append((layer.weights, layer.biases))
        with open(path, "wb") as f:
            pickle.dump(weights, f)

    def load_weights(self, path):
        with open(path, "rb") as f:
            weights = pickle.load(f)
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.weights, layer.biases = weights[idx]
                idx += 1
