from layer import Layer
from activation_function import ActivationFunction

# Layer applying the sigmoid activation function
class SigmoidLayer(Layer):
    # Forward pass applying sigmoid function
    def forward(self, inputs):
        self.input = inputs
        self.output = ActivationFunction.sigmoid(inputs)

    # Backward pass for the sigmoid layer
    def backward(self, output_gradient):
        return output_gradient * ActivationFunction.sigmoid_derivative(self.output)
