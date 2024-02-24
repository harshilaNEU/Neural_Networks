# HW to Chapter 8 “Parameter Initialization and Training Sets”

## Name : Harshila Jagtap 

NEU ID : 002743674 

## Question :

Develop a prototype of the training, validation, and testing sets of your choice for the future training of your neural network. The term “prototype: means that the sets may be quite limited by number of images, but the proportions of images in them should be maintained as required

## Pre-requisite :

1. Visual Studio Code
2. Python

## Dataset
CIFAR-10 is a dataset consisting of 60000, 32×32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. More details about the datset can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

Sample images from each class of the CIFAR-10 dataset is shown below:



![Dataset](https://github.com/harshilaNEU/Neural_Networks/blob/Training_Sets/Reference_Images/CIFAR-10_dataset.png)

## To Generate the Prototype

1. Took 10% of the of Train dataset, 1% of each class 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'. So considered 5000 images.
2. Took 10% of the Test dataset, 1% of each class 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'. So considered 1000 images.
3. For Validation, setting 20% aside and 80% for training.

## Downloaded CIFAR-10 dataset

![Original CIFAR-10 dataset](https://github.com/harshilaNEU/Neural_Networks/blob/Training_Sets/Reference_Images/Downloaded_CIFAR-10_data.png)

## Output of Generated Prototype

![Generated_Prototype](https://github.com/harshilaNEU/Neural_Networks/blob/Training_Sets/Reference_Images/output.png)


![Classwise_prototype_generation](https://github.com/harshilaNEU/Neural_Networks/blob/Training_Sets/Reference_Images/internal_folder_structure.png)
