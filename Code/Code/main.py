from neural_network import NeuralNetwork
from dense_layer import DenseLayer
from sigmoid_layer import SigmoidLayer
from loss_functions import mean_squared_error, mean_squared_error_derivative
import numpy as np

# Create the neural network
nn = NeuralNetwork()
# Input layer : 2 inputs, 4 neurons
nn.add(DenseLayer(2, 4))
# Sigmoid Activation Layer
nn.add(SigmoidLayer())
# Hidden layer : 4 neurons, 1 output
nn.add(DenseLayer(4, 1))
# Sigmoid Activation Layer
nn.add(SigmoidLayer())

# Training data for the XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn.fit(X, y, epochs=1000, learning_rate=0.1, loss=mean_squared_error, loss_derivative=mean_squared_error_derivative)
