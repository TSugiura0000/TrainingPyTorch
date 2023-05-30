import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_mnist_dataloader():
    # setting
    data_path = './datasets/MNIST_data'

    # load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ]
    )
    train_set = datasets.MNIST(
        data_path, download=True, train=True, transform=transform
    )
    test_set = datasets.MNIST(
        data_path, download=True, train=False, transform=transform
    )

    # create dataloader
    b_size = 256
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=b_size, shuffle=True)
    return train_loader, test_loader


class Autoencoder(nn.Module):
    def __init__(
            self,
            input_dim: tuple,
            encoder_conv_channels: list,
            encoder_conv_kernel_size: list,
            encoder_conv_strides: list,
            decoder_conv_t_channels: list,
            decoder_conv_t_kernel_size: list,
            decoder_conv_t_strides: list,
            z_dim: int = 3
    ) -> None:
        if not (len(encoder_conv_channels)
                == len(encoder_conv_kernel_size)
                == len(encoder_conv_strides)):
            raise ValueError(
                "List size of 'encoder_conv_filters',"
                " 'encoder_conv_kernel_size', and 'encoder_conv_strides' "
                "must be same."
            )
        if not (len(decoder_conv_t_channels)
                == len(decoder_conv_t_kernel_size)
                == len(decoder_conv_t_strides)):
            raise ValueError(
                "List size of 'decoder_conv_t_filters', "
                "'decoder_conv_t_kernel_size', and 'decoder_conv_t_strides' "
                "must be same."
            )
        super().__init__()
        self._input_dim = input_dim
        self._encoder_conv_channels = encoder_conv_channels
        self._encoder_conv_kernel_size = encoder_conv_kernel_size
        self._encoder_conv_strides = encoder_conv_strides
        self._decoder_conv_t_channels = decoder_conv_t_channels
        self._decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self._decoder_conv_t_strides = decoder_conv_t_strides
        self._z_dim = z_dim

        # Encoder
        encoder_conv_layers = []
        out_height = input_dim[1]
        out_width = input_dim[2]
        pad = 1
        dil = 1
        for i, k_size, stride in enumerate(zip(
            self._encoder_conv_kernel_size,
            self._encoder_conv_strides
        )):
            if i == 0:
                input_channels = self._input_dim[0]
            else:
                input_channels = self._encoder_conv_channels[i - 1]
            encoder_conv_layers.append(
                nn.Conv2d(
                    input_channels,
                    self._encoder_conv_channels[i],
                    kernel_size=k_size,
                    stride=stride,
                    padding=pad
                )
            )
            out_height = np.floor(
                (out_height + 2 * pad - dil * (k_size - 1) - 1) / stride + 1
            )
            out_width = np.floor(
                (out_width + 2 * pad - dil * (k_size - 1) - 1) / stride + 1
            )
        self._encoder_conv_layers = nn.ModuleList(encoder_conv_layers)
        self._input_encode_linear = \
            self._encoder_conv_channels[-1] * out_height * out_width
        self._pre_reshape_height_of_encoder = out_height
        self._pre_reshape_width_of_encoder = out_width
        self._encoder_output = nn.Linear(
            self._input_encode_linear,
            self._z_dim
        )

        # Decoder
        self._output_decode_linear = self._input_encode_linear
        self._decoder_input_linear = nn.Linear(
            z_dim,
            self._output_decode_linear
        )
        decoder_conv_t_layers = []
        out_height = self._pre_reshape_height_of_encoder
        out_width = self._pre_reshape_width_of_encoder
        pad = 1
        dil = 1
        out_pad = 1
        for i, k_size, stride in enumerate(zip(
            self._decoder_conv_t_kernel_size,
            self._decoder_conv_t_strides
        )):
            if i == 0:
                input_conv_t = self._decoder_conv_t_channels[0]
            else:
                input_conv_t = self._decoder_conv_t_channels[i - 1]
            decoder_conv_t_layers.append(
                nn.ConvTranspose2d(
                    input_conv_t,
                    self._decoder_conv_t_channels[i],
                    kernel_size=k_size,
                    stride=stride,
                    padding=pad,
                    output_padding=out_pad
                )
            )
            out_height = \
                (out_height - 1) * stride - 2 * pad + dil * (k_size - 1) \
                + out_pad + 1
            out_width = \
                (out_width - 1) * stride - 2 * pad + dil * (k_size - 1) \
                + out_pad + 1
        self._decoder_conv_t_layers = nn.ModuleList(decoder_conv_t_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        for conv in self._encoder_conv_layers:
            x = conv(x)
            x = F.relu(x)
        x = x.view(
            -1,
            self._input_encode_linear
        )
        x = self._encoder_output(x)

        # Decoder
        x = self._decoder_input_linear(x)
        x = x.view(
            -1,
            self._decoder_conv_t_channels[0],
            self._pre_reshape_height_of_encoder,
            self._pre_reshape_width_of_encoder
        )
        for conv_t in self._decoder_conv_t_layers:
            x = conv_t(x)
            x = F.relu(x)


