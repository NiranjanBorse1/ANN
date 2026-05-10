# Back Propagation Network for XOR Function

import numpy as np

# Binary Input
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# Binary Output (XOR)
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights
input_hidden_weights = np.random.rand(2,2)
hidden_output_weights = np.random.rand(2,1)

# Learning rate
lr = 0.5

# Training using Back Propagation
for epoch in range(1000):

    # Forward Propagation

    hidden_input = np.dot(X, input_hidden_weights)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, hidden_output_weights)
    final_output = sigmoid(final_input)

    # Error Calculation
    error = Y - final_output

    # Back Propagation

    d_output = error * sigmoid_derivative(final_output)

    hidden_error = d_output.dot(hidden_output_weights.T)

    d_hidden = hidden_error * sigmoid_derivative(hidden_output)

    # Weight Update

    hidden_output_weights += hidden_output.T.dot(d_output) * lr

    input_hidden_weights += X.T.dot(d_hidden) * lr

# Final Output
print("Output after Training:")
print(np.round(final_output))