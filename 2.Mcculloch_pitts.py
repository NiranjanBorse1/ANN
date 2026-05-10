# Inputs: (x1, x2)
inputs = [(0,0), (0,1), (1,0), (1,1)]

print("--- Strict M-P Model (Inhibitory Veto) ---")
print("x1  x2  Output")

for x1, x2 in inputs:
    # M-P Rule: 
    # x1 = Excitatory
    # x2 = Inhibitory
    theta = 1
    
    # Logic: If inhibitory input is 1, output is 0. 
    # Otherwise, check if sum of excitatory inputs >= theta.
    if x2 == 1:
        y = 0
    elif x1 >= theta:
        y = 1
    else:
        y = 0
        
    print(f"{x1}   {x2}     {y}")

# McCulloch-Pitts Neural Network for AND Function

# Input combinations
'''inputs = [(0,0), (0,1), (1,0), (1,1)]

print("x1  x2  Output")

for x1, x2 in inputs:

    # Threshold
    theta = 2

    # Net input
    net = x1 + x2

    # Activation function
    if net >= theta:
        y = 1
    else:
        y = 0

    print(f"{x1}   {x2}     {y}")'''