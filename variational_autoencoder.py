from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_mnist_dataloader(n_batch: int = 256):
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
    train_loader = DataLoader(train_set, batch_size=n_batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=n_batch, shuffle=True)
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
        for i, (k_size, stride) in enumerate(
            zip(
                self._encoder_conv_kernel_size,
                self._encoder_conv_strides
            )
        ):
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
            out_height = int(np.floor(
                (out_height + 2 * pad - dil * (k_size - 1) - 1) / stride + 1
            ))
            out_width = int(np.floor(
                (out_width + 2 * pad - dil * (k_size - 1) - 1) / stride + 1
            ))
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
        output_padding = [st - 1 for st in self._decoder_conv_t_strides]
        for i, (k_size, stride, out_pad) in enumerate(zip(
                self._decoder_conv_t_kernel_size,
                self._decoder_conv_t_strides,
                output_padding
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
        for i, conv_t in enumerate(self._decoder_conv_t_layers):
            x = conv_t(x)
            if i < len(self._decoder_conv_t_layers) - 1:
                x = F.relu(x)
            else:
                x = F.sigmoid(x)
        return x


class RMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, reconstructed_x, x) -> torch.Tensor:
        return torch.sqrt(self.mse(reconstructed_x, x))


if __name__ == '__main__':
    # setting device
    device = torch.device('cuda') \
        if torch.cuda.is_available() else torch.device('cpu')
    print('Device: ', device)

    # load dataloader
    print('Loading DataLoader . . .')
    train_dataloader, test_dataloader = create_mnist_dataloader()
    print('Completed Loading DataLoader\n')

    # create model and move to 'device'
    model = Autoencoder(
        input_dim=(1, 28, 28),
        encoder_conv_channels=[32, 64, 64, 64],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[1, 2, 2, 1],
        decoder_conv_t_channels=[64, 64, 32, 1],
        decoder_conv_t_kernel_size=[3, 3, 3, 3],
        decoder_conv_t_strides=[1, 2, 2, 1],
        z_dim=10
    )
    model.to(device=device)

    # loss function
    loss_fn = RMSELoss()

    # learning rate
    lr = 1e-2

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # training setting
    n_epochs = 10
    step = 1

    train_losses = []
    test_losses = []

    print('====== Start training and test ======')
    start_time = datetime.now()
    for epoch in range(1, n_epochs + 1):
        # training
        model.train()
        loss_train = 0.0
        for images, _ in train_dataloader:
            images = images.to(device)
            batch_size = images.shape[0]
            reconstructed_images = model(images)
            loss = loss_fn(reconstructed_images, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch % step == 0:
            train_losses.append(loss_train / len(train_dataloader))

        # test
        model.eval()
        loss_test = 0.0

        with torch.no_grad():
            for images, _ in test_dataloader:
                images = images.to(device)
                batch_size = images.shape[0]
                reconstructed_images = model(images)
                loss = loss_fn(reconstructed_images, images)
                loss_test += loss.item()

            if epoch % step == 0:
                test_losses.append(loss_test / len(test_dataloader))

        if epoch % step == 0 or epoch == 1:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(
                'Time: {}, Epoch: {}, Training oss: {}, Test loss: {}'.format(
                    current_time,
                    epoch,
                    float(train_losses[-1]),
                    float(test_losses[-1])
                )
            )
    end_time = datetime.now()
    print('training time: ', end_time - start_time)

    # plot
    train_loss = np.array(train_losses)
    test_loss = np.array(test_losses)

    figure = plt.figure(figsize=(15, 7))
    axis = figure.add_subplot(111)
    axis.plot(
        [i * step for i in range(1, n_epochs // step + 1)],
        train_loss, label='Train loss'
    )
    axis.plot(
        [i * step for i in range(1, n_epochs // step + 1)],
        test_loss, label='Test loss'
    )
    axis.set_title('Autoencoder')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Loss')
    axis.legend()

    save_path_as_pdf = '.result/mnist_autoencoder_sgd.pdf'
    save_path_as_png = '.result/mnist_autoencoder_sgd.png'
    plt.savefig(save_path_as_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_path_as_png, format='PNG', bbox_inches='tight')
    plt.show()
