class Trainer:
    # Initialize the Trainer with a model, dataset, and corresponding labels
    def __init__(self, model, dataset, labels):
        # Model to be trained
        self.model = model
        # Dataset for training
        self.dataset = dataset
        # Corresponding labels for the dataset
        self.labels = labels

    # Train the model using the provided dataset and labels
    def train(self, epochs, learning_rate):
        self.model.train(self.dataset, self.labels, epochs, learning_rate)

class Tester:
    # Initialize the Tester with a model and a dataset
    def __init__(self, model, dataset):
        # Model to be tested
        self.model = model
        # Dataset for testing
        self.dataset = dataset

    # Test the model using the provided dataset
    def test(self):
        predictions = [self.model.predict(data) for data in self.dataset]
        # Return the list of predictions
        return predictions
