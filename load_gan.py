import os

import torch
from torch import nn

from gan import Discriminator, Generator


# setting device
device = torch.device('cuda') \
    if torch.cuda.is_available() else torch.device('cpu')


def plot_generated_image(generator: nn.Module, z: torch.Tensor):
    dir_path = './'


if __name__ == '__main__':
    # setting model
    z_dim = 100
    d = Discriminator(
        input_dim=(1, 28, 28),
        conv_kernel_size=[5, 5, 5, 5],
        conv_kernel_filter=[64, 64, 128, 128],
        dropout_rate=0.4,
        activation='relu'
    )
    g = Generator(
        z_dim=z_dim,
        init_linear_size=(64, 7, 7),
        conv_kernel_size=[5, 5, 5, 5],
        conv_kernel_filter=[128, 64, 64, 1],
        dropout_rate=0.0,
        batch_norm=True,
        activation='relu',
        output_size=(1, 28, 28)
    )
    d.to(device=device)
    g.to(device=device)

    # load state dictionary
    model_dir = './model'
    d.load_state_dict(torch.load(os.path.join(model_dir, 'discriminator.pth')))
    g.load_state_dict(torch.load(os.path.join(model_dir, 'generator.pth')))


