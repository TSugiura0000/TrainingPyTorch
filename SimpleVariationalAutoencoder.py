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

