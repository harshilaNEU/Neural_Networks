from neuron import Neuron
from activation_function import ActivationFunction

class Perceptron(Neuron):
    # Initialize Perceptron
    def __init__(self, input_size):
        # Initialize the base Neuron class
        super().__init__(input_size)
        # Set the activation function
        self.activation_function = ActivationFunction.sigmoid
        # Set the derivative of the activation function
        self.derivative = ActivationFunction.sigmoid_derivative

    # Train the Perceptron over a specified number of epochs
    def train(self, inputs, label, epochs, learning_rate):
        for _ in range(epochs):
            for input_vector, label_vector in zip(inputs, label):
                # Compute the predicted output
                output = self.activation_function(self.output(input_vector))
                # Calculate the error
                error = label_vector - output
                # Compute the derivative of the output
                derivative = self.derivative(output)
                # Compute and return the predicted output
                self.adjust_weights(input_vector, error * derivative, learning_rate)

    # Predict the output for a given input vector using the Perceptron
    def predict(self, input_vector):
        return self.activation_function(self.output(input_vector))
