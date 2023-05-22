import torch.nn as nn

# Get all the classes in nn module
nn_classes = [attr for attr in dir(nn) if not attr.startswith('__')]

# Filter only the loss functions
loss_functions = [cls for cls in nn_classes if 'Loss' in cls]

# Print the loss functions
for loss in loss_functions:
    print(loss)
