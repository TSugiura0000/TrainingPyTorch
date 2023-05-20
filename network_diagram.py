import torch
from torch import nn
from torchviz import make_dot


# Define the model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
)

# Create a random tensor that has the same size as the input layer
x = torch.randn(1, 784).requires_grad_(True)

# Forward pass through the model
y = model(x)

# Create the visualization
dot = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
dot.format = 'pdf'
dot.render('network_graph')
