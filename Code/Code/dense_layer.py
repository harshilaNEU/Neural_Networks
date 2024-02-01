import numpy as np
from layer import Layer

# Dense Layer
class DenseLayer(Layer):
    # Initialize weights and biases
    def __init__(self, n_inputs, n_neurons):
        super().__init__()
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

    # Forward propagation through dense layer
    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward propagation through dense layer
    def backward(self, output_gradient):
        self.weight_gradient = np.dot(self.input.T, output_gradient)
        self.bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.weights.T)
