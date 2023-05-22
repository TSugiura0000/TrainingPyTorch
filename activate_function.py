import torch
import matplotlib.pyplot as plt

# Prepare the input data
x = torch.linspace(-5, 5, 200)

# Compute the activation functions
activations = [
    ('Tanh', torch.tanh(x)),
    ('HardTanh', torch.nn.functional.hardtanh(x)),
    ('Sigmoid', torch.sigmoid(x)),
    ('Softplus', torch.nn.functional.softplus(x)),
    ('ReLU', torch.nn.functional.relu(x)),
    ('Leaky ReLU', torch.nn.functional.leaky_relu(x))
]

# Setup the figure and axes
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()  # flatten the 2D array of axes

# Plot the activation functions
for i, (name, y) in enumerate(activations):
    ax = axes[i]
    ax.plot(x.numpy(), y.numpy())
    ax.set_title(name)
    ax.grid(True)

plt.tight_layout()
plt.show()
