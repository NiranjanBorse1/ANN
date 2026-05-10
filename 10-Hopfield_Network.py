# Hopfield Network to Store 4 Vectors

import numpy as np

# 4 bipolar vectors
v1 = np.array([1, -1, 1, -1])
v2 = np.array([-1, 1, -1, 1])
v3 = np.array([1, 1, -1, -1])
v4 = np.array([-1, -1, 1, 1])

# Store vectors in list
vectors = [v1, v2, v3, v4]

# Initialize weight matrix
W = np.zeros((4,4))

# Training Hopfield Network
for v in vectors:

    # Outer product
    W = W + np.outer(v, v)

# Remove self connections
np.fill_diagonal(W, 0)

# Print weight matrix
print("Weight Matrix:")
print(W)