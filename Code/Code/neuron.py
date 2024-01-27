class Neuron:
    # Initialize the Neuron with a specified input size with additional bias weight
    def __init__(self, input_size):
        self.weights = [0.01] * (input_size + 1)

    # Calculate the weighted sum of the inputs and bias
    def output(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights[1:], inputs)) + self.weights[0]
        return weighted_sum

    # Adjust the weights of the neuron based on the error and learning rate
    def adjust_weights(self, inputs, error, learning_rate):
        for i in range(len(inputs)):
            self.weights[i + 1] += learning_rate * error * inputs[i]
        self.weights[0] += learning_rate * error
