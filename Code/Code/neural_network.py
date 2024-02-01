# Neural Network Model
class NeuralNetwork:
    # Initialize the neural network with an empty list of layers
    def __init__(self):
        self.layers = []

    # Add a layer to the neural network
    def add(self, layer):
        self.layers.append(layer)

    # Compute output of the network for given inputs
    def predict(self, X):
        output = X
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    # Perform backpropagation, computing gradients for training
    def backward(self, y_true, y_pred, loss_derivative):
        loss_gradient = loss_derivative(y_true, y_pred)
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)

    # Train the neural network using the provided training data
    def fit(self, X, y, epochs, learning_rate, loss, loss_derivative):
        for epoch in range(epochs):
            y_pred = self.predict(X)
            loss_value = loss(y, y_pred)
            self.backward(y, y_pred, loss_derivative)
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    layer.weights -= learning_rate * layer.weight_gradient
                    layer.biases -= learning_rate * layer.bias_gradient
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss_value}')
