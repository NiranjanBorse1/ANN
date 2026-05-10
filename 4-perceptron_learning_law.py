# Perceptron Learning Law with Decision Region
import matplotlib.pyplot as plt

# AND gate inputs
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]

# Target output
target = [0, 0, 0, 1]

# Initial weights and bias
w1 = 0
w2 = 0
b = 0

# Learning rate
lr = 0.1

# Training perceptron
for epoch in range(5):

    for i in range(4):

        # Net input
        net = x1[i] * w1 + x2[i] * w2 + b

        # Activation function
        if net >= 0:
            output = 1
        else:
            output = 0

        # Error calculation
        error = target[i] - output

        # Weight update
        w1 = w1 + lr * error * x1[i]
        w2 = w2 + lr * error * x2[i]

        # Bias update
        b = b + lr * error

# Print final values
print("Final Weight1 =", w1)
print("Final Weight2 =", w2)
print("Final Bias =", b)

# Plot input points
for i in range(4):

    if target[i] == 0:
        plt.scatter(x1[i], x2[i], marker='o')
    else:
        plt.scatter(x1[i], x2[i], marker='x')

# Decision boundary
x = [-1, 2]

y = [(-(w1 * i) - b) / w2 for i in x]

plt.plot(x, y)

# Graph labels
plt.title("Perceptron Decision Region")
plt.xlabel("x1")
plt.ylabel("x2")

plt.grid(True)

# Show graph
plt.show()