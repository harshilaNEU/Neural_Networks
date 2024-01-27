import numpy as np

class ActivationFunction:
    # Compute the Sigmoid Activation Function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Compute the derivative of the sigmoid function.
    def sigmoid_derivative(output):
        return output * (1 - output)
