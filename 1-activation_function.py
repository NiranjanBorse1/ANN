# Python program to plot different activation functions
# in separate figures
import numpy as np
import matplotlib.pyplot as plt

# Input values
x = np.linspace(-10, 10, 100)

# 1. Sigmoid Function
y1 = 1 / (1 + np.exp(-x))

plt.figure()
plt.plot(x, y1)
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# 2. Tanh Function
y2 = np.tanh(x)

plt.figure()
plt.plot(x, y2)
plt.title("Tanh Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# 3. ReLU Function
y3 = np.maximum(0, x)

plt.figure()
plt.plot(x, y3)
plt.title("ReLU Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# 4. Leaky ReLU Function
y4 = np.where(x > 0, x, 0.01 * x)

plt.figure()
plt.plot(x, y4)
plt.title("Leaky ReLU Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# Show all figures
plt.show()