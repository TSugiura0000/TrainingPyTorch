import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary

# setting device
device = torch.device('cuda') \
    if torch.cuda.is_available() else torch.device('cpu')


def create_mnist_dataloader(n_batch: int = 256):
    # setting
    data_path = './datasets/MNIST_data'

    # load dataset
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    train_set = datasets.MNIST(
        data_path, download=True, train=True, transform=transform
    )
    test_set = datasets.MNIST(
        data_path, download=True, train=False, transform=transform
    )

    # create dataloader
    train_loader = DataLoader(train_set, batch_size=n_batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=n_batch, shuffle=True)
    return train_loader, test_loader


class Discriminator(nn.Module):
    def __init__(
            self,
            input_dim: tuple = (1, 28, 28),
            conv_kernel_size: None or list = None,
            conv_kernel_filter: None or list = None,
            dropout_rate: float = 0.0,
            activation: str = 'relu'
    ) -> None:
        """Discriminator
        The number of elements in the list
        (conv_kernel_size, conv_kernel_filter) must be 4.
        """
        super().__init__()
        dropout_layer = nn.Dropout2d(p=dropout_rate)
        activate_fn = nn.ReLU()
        if conv_kernel_size is None:
            conv_kernel_size = [3, 3, 3, 3]
        if conv_kernel_filter is None:
            conv_kernel_filter = [64, 64, 128, 128]
        if activation == 'leaky_relu':
            activate_fn = nn.LeakyReLU(inplace=True)

        self._input_dim = input_dim
        self._conv_kernel_size = conv_kernel_size
        self._conv_kernel_filter = conv_kernel_filter
        self._dropout = dropout_layer
        self._activate_fn = activate_fn
        self._conv_layers = nn.ModuleList(
            [nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=k_size,
                stride=1,
                padding=1
            )
                for in_chan, out_chan, k_size
                in zip(
                [self._input_dim[0]] + self._conv_kernel_filter[0:3],
                self._conv_kernel_filter,
                self._conv_kernel_size
            )]
        )
        self._last_linear = nn.Linear(
            ((self._input_dim[1] // 8) ** 2) * conv_kernel_filter[-1],
            1
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        for i, conv in enumerate(self._conv_layers):
            x = self._activate_fn(conv(x))
            x = self._dropout(x)
            if i != len(self._conv_layers) - 1:
                x = F.max_pool2d(x, 2)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self._last_linear(x)
        x = torch.sigmoid(x)
        loss = F.binary_cross_entropy(x, y)
        return x, loss


class Generator(nn.Module):
    def __init__(
            self,
            z_dim: int = 100,
            init_linear_size: tuple = (64, 7, 7),
            conv_kernel_size: None or list = None,
            conv_kernel_filter: None or list = None,
            dropout_rate: float = 0.0,
            batch_norm: bool = True,
            activation: str = 'relu',
            output_size: tuple = (1, 28, 28)
    ) -> None:
        """Discriminator
        The number of elements in the list
        (conv_kernel_size, conv_kernel_filter) must be 4.
        """
        super().__init__()
        batch_norm_layers = []
        dropout_layer = nn.Dropout2d(p=dropout_rate)
        activate_fn = nn.ReLU()
        init_linear_features = \
            init_linear_size[0] * init_linear_size[1] * init_linear_size[2]
        if conv_kernel_size is None:
            conv_kernel_size = [3, 3, 3, 3]
        if conv_kernel_filter is None:
            conv_kernel_filter = [128, 64, 64, 1]
        if activation == 'leaky_relu':
            activate_fn = nn.LeakyReLU(inplace=True)
        if batch_norm:
            batch_norm_layers.append(nn.BatchNorm1d(init_linear_features))
            for features in conv_kernel_filter[0:-1]:
                batch_norm_layers.append(nn.BatchNorm2d(features))
        else:
            for _ in range(4):
                batch_norm_layers.append(nn.Identity())

        self._z_dim = z_dim
        self._output_size = output_size
        self._conv_kernel_size = conv_kernel_size
        self._conv_kernel_filter = conv_kernel_filter
        self._init_linear_size = init_linear_size
        self._init_linear = nn.Linear(self._z_dim, init_linear_features)
        self._nn_batch_norm_layers = nn.ModuleList(batch_norm_layers)
        self._dropout = dropout_layer
        self._activate_fn = activate_fn
        self._conv_layers = nn.ModuleList(
            [nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=k_size,
                stride=1,
                padding=1
            )
                for in_chan, out_chan, k_size
                in zip(
                [self._init_linear_size[0]] + self._conv_kernel_filter[0:3],
                self._conv_kernel_filter,
                self._conv_kernel_size
            )]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self._init_linear(z)
        x = self._nn_batch_norm_layers[0](x)
        x = self._activate_fn(x)
        x = x.view(-1,
                   self._init_linear_size[0],
                   self._init_linear_size[1],
                   self._init_linear_size[2],
                   )
        for i, conv in enumerate(self._conv_layers):
            if i < 2:
                x = F.interpolate(
                    x, scale_factor=2, mode='bilinear', align_corners=False
                )
            x = conv(x)
            if i != len(self._nn_batch_norm_layers):
                x = self._nn_batch_norm_layers[i + 1](x)
            x = self._activate_fn(x)
        return x




if __name__ == '__main__':
    print('Device: ', device)

    # load dataloader
    print('Loading DataLoader . . .')
    train_dataloader, test_dataloader = create_mnist_dataloader(n_batch=128)
    print('Completed Loading DataLoader\n')
