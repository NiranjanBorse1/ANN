# Simple Perceptron Program for Even and Odd Numbers
# ASCII values of 0 to 9
numbers = [48,49,50,51,52,53,54,55,56,57]

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
        print(chr(x), "       Odd")