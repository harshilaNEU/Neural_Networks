import sys
sys.path.append('Code')
sys.path.append('Data')
from perceptron import Perceptron
from dataset import Dataset
from train_test import Trainer, Tester

# Set Path to train and test image datasets
train_folder = '../Data/Train_Images'
test_folder = '../Data/Test_Images'
input_size = 400  # Since we used 20x20 images

# Load datasets
train_data = Dataset(train_folder).load_images()
test_data = Dataset(test_folder).load_images()

# Matrix for each digit 10 labels starting from 0 to 9
train_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9]


# Initialize the Perceptron
perceptron = Perceptron(input_size)

# Train the Perceptron
trainer = Trainer(perceptron, train_data, train_labels)
trainer.train(epochs=50, learning_rate=0.01)

# Test the Perceptron
tester = Tester(perceptron, test_data)
predictions = tester.test()

# Display the Result
print("Predictions:", predictions)
