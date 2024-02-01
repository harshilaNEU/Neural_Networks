import numpy as np

# Mean squared error loass function
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Derivative of the mean squared error loss function
def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
