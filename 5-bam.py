# Bidirectional Associative Memory (BAM)
import numpy as np

# Input vectors
X1 = np.array([1, -1, 1])
X2 = np.array([-1, 1, -1])

# Output vectors
Y1 = np.array([1, -1])
Y2 = np.array([-1, 1])

# Weight matrix initialization
W = np.zeros((3,2))

# Training BAM Network
W = W + np.outer(X1, Y1)
W = W + np.outer(X2, Y2)

# Print weight matrix
print("Weight Matrix:")
print(W)

# Testing
print("\nTesting:")

test = X1

# Forward association
Y = np.dot(test, W)

# Activation function
Y = np.where(Y >= 0, 1, -1)

print("Input Vector :", test)
print("Associated Output :", Y)