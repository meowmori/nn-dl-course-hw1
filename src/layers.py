"""
实现神经网络层和激活函数
"""

import numpy as np


class Layer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros(output_dim)
        self.inputs = None
        self.gradients = {}

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, grad_output):
        self.gradients["weights"] = np.dot(self.inputs.T, grad_output)
        self.gradients["biases"] = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.weights.T)


class ReLU(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        return grad_output * (self.inputs > 0)


class Sigmoid(Layer):
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)


class Tanh(Layer):
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output**2)
