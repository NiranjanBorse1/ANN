# Simple Perceptron Program for Even and Odd Numbers
# ASCII values of 0 to 9
'''numbers = [48,49,50,51,52,53,54,55,56,57]

# Weight and bias
w = 1
b = 0

print("Number   Result")

for x in numbers:

    # Net input
    net = x * w + b

    # Activation function
    if net % 2 == 0:
        output = 0
    else:
        output = 1

    # Final output
    if output == 0:
        print(chr(x), "       Even")
    else:
        print(chr(x), "       Odd")'''


import numpy as np

# 1. Prepare Data (Numbers 0-9 in ASCII)
numbers = np.array(range(48, 58))  # ASCII 48 to 57

# Convert ASCII integers to 8-bit binary lists
# Example: 48 -> [0, 0, 1, 1, 0, 0, 0, 0]
X = np.array([list(map(int, format(n, '08b'))) for n in numbers])

# Labels: 0 for Even, 1 for Odd
# 48 is Even(0), 49 is Odd(1), etc.
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# 2. Initialize Perceptron Parameters
weights = np.zeros(8) # One weight for each bit in the ASCII byte
bias = 0
lr = 0.1              # Learning Rate
epochs = 50           # Iterations

# 3. Training (The Learning Process)
for epoch in range(epochs):
    for i in range(len(X)):
        # Linear Combination (Net Input)
        net_input = np.dot(X[i], weights) + bias
        
        # Step Activation Function
        prediction = 1 if net_input > 0 else 0
        
        # Perceptron Update Rule: W = W + lr * (target - prediction) * input
        error = y[i] - prediction
        weights += lr * error * X[i]
        bias += lr * error

# 4. Results and Testing
print("--- Training Complete ---")
print(f"Final Weights: {weights}")
print(f"Final Bias: {bias}")
print("-" * 40)
print(f"{'Char':<6} {'ASCII':<8} {'Binary':<12} {'Result':<10}")

for i, val in enumerate(numbers):
    # Predict using the trained weights
    bits = X[i]
    z = np.dot(bits, weights) + bias
    prediction = 1 if z > 0 else 0
    
    label = "Odd" if prediction == 1 else "Even"
    print(f"{chr(val):<6} {val:<8} {str(bits):<12} {label:<10}")