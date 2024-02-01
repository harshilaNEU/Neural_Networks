# Base class for layers for the neural network
class Layer:
    def __init__(self):
        self.output = None
        self.input = None

    # Forward pass through the layer
    def forward(self, inputs):
        raise NotImplementedError

    # Forward pass through the layer
    def backward(self, output_gradient):
        raise NotImplementedError
