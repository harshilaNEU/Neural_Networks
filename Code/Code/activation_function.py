import numpy as np

# Activation functions and its derivatives
class ActivationFunction:
    # Sigmoid activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Derivative of sigmoid function
    def sigmoid_derivative(x):
        return x * (1 - x)
