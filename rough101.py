import numpy as np
import matplotlib.pyplot as plt

# Define the function (Loss function)
def loss_function(x):
    return x**2 + 2*x + 1

# Define the gradient of the loss function
def gradient(x):
    return 2*x + 2

# Generate data points
x_values = np.linspace(-5, 5, 400)  # Range for x-axis
loss_values = loss_function(x_values)  # Calculate the loss at each x
gradient_values = gradient(x_values)  # Calculate the gradient at each x

# Plotting
plt.figure(figsize=(10, 6))

# Plot the loss function
plt.plot(x_values, loss_values, label="Loss Function", color="blue", linewidth=2)

# Plot the gradient
plt.plot(x_values, gradient_values, label="Gradient", color="red", linestyle='--', linewidth=2)

plt.title("Loss Function and Gradient")
plt.xlabel("x")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
