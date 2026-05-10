# Back Propagation Feed Forward Neural Network

import numpy as np

# Input data
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# Target output
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights
input_hidden_weights = np.random.rand(2,2)
hidden_output_weights = np.random.rand(2,1)

# Learning rate
lr = 0.5

# Training network
for epoch in range(10000):

    # Forward Propagation

    hidden_input = np.dot(X, input_hidden_weights)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, hidden_output_weights)
    final_output = sigmoid(final_input)

    # Error calculation
    error = Y - final_output

    # Back Propagation

    d_output = error * sigmoid_derivative(final_output)

    hidden_error = d_output.dot(hidden_output_weights.T)

    d_hidden = hidden_error * sigmoid_derivative(hidden_output)

    # Weight update

    hidden_output_weights += hidden_output.T.dot(d_output) * lr

    input_hidden_weights += X.T.dot(d_hidden) * lr

# Final Output
print("Final Output:")
print(np.round(final_output))