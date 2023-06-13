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


def plot_comparison(x: torch.Tensor, re_x: torch.Tensor,
                    epoch_num: int, n: int = 10) -> None:
    dir_path = './comparison_images/'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    x = x.cpu().numpy()
    re_x = re_x.cpu().detach().numpy()

    x = 255 / (np.max(x) - np.min(x)) * x - np.min(x)
    re_x = 255 / (np.max(re_x) - np.min(re_x)) * re_x - np.min(re_x)

    fig = plt.figure(figsize=(20, 4))
    gs = gridspec.GridSpec(2, n)
    gs.update(wspace=0.05, hspace=0.05)

    for i in range(n):
        # plot original image
        ax = plt.subplot(gs[0, i])
        plt.imshow(x[i].reshape(28, 28), cmap='gray')
        ax.axis('off')

        # plot reconstructed image
        ax = plt.subplot(gs[1, i])
        plt.imshow(re_x[i].reshape(28, 28), cmap='gray')
        ax.axis('off')

    plt.savefig(
        f'./comparison_images/epoch_{epoch_num}.pdf', bbox_inches='tight'
    )
    plt.close(fig)


class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            input_dim: tuple,
            encoder_conv_channels: list,
            encoder_conv_kernel_size: list,
            encoder_conv_strides: list,
            decoder_conv_t_channels: list,
            decoder_conv_t_kernel_size: list,
            decoder_conv_t_strides: list,
            z_dim: int = 3,
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
        self._mu = nn.Linear(
            self._input_encode_linear,
            self._z_dim
        )
        self._log_var = nn.Linear(
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

    def forward(self, x: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder
        for conv in self._encoder_conv_layers:
            x = conv(x)
            x = F.leaky_relu(x)
        x = x.view(
            -1,
            self._input_encode_linear
        )
        _mu = self._mu(x)
        _log_var = self._log_var(x)
        x = VariationalAutoencoder.sampling(_mu, _log_var)

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
                x = F.leaky_relu(x)
            else:
                x = F.sigmoid(x)
        return x, _mu, _log_var

    @staticmethod
    def sampling(mu_: torch.Tensor, log_var_: torch.Tensor) -> torch.Tensor:
        std = torch.exp(log_var_ / 2)
        epsilon = torch.randn_like(std)
        return mu_ + epsilon * std

    @staticmethod
    def loss_function(recon_x, x, mu_, log_var_):
        bce = F.mse_loss(recon_x, x)
        kld = -0.5 * torch.sum(
            1 + log_var_ - torch.square(mu_) - torch.exp(log_var_)
        )
        return bce + kld


class SimpleVAE(nn.Module):
    def __init__(self, input_dim: tuple, z_dim: int = 64):
        super(SimpleVAE, self).__init__()
        self._input_dim = input_dim
        self._input_channels = input_dim[0]
        self._input_height = input_dim[1]
        self._input_width = input_dim[2]
        self._z_dim = z_dim

        # Encoder
        self._encoder_linear1 = nn.Linear(
            self._input_channels * self._input_height * self._input_width,
            512
        )
        self._encoder_linear2 = nn.Linear(512, 64)
        self._mu = nn.Linear(64, self._z_dim)
        self._log_var = nn.Linear(64, self._z_dim)

        # Decoder
        self._decoder_linear1 = nn.Linear(self._z_dim, 64)
        self._decoder_linear2 = nn.Linear(64, 512)
        self._decoder_linear3 = nn.Linear(
            512,
            self._input_channels * self._input_height * self._input_width
        )

    def encoder(self, x):
        x = F.relu(self._encoder_linear1(x))
        x = F.relu(self._encoder_linear2(x))
        mu_ = self._mu(x)
        log_var_ = self._log_var(x)
        return x, mu_, log_var_

    def decoder(self, z):
        x = F.relu(self._decoder_linear1(z))
        x = F.relu(self._decoder_linear2(x))
        x = F.sigmoid(self._decoder_linear3(x))
        return x

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x, mu_, log_var_ = self.encoder(x)
        z = self.sampling(mu_, log_var_)
        x = self.decoder(z)
        x = x.reshape(
            (
                -1,
                self._input_channels,
                self._input_height,
                self._input_width
            )
        )
        return x, z, mu_, log_var_

    def sampling(self, mu_: torch.Tensor, log_var_: torch.Tensor) \
            -> torch.Tensor:
        std = torch.exp(log_var_ / 2)
        epsilon = torch.randn(mu_.shape, device=device)
        return mu_ + epsilon * std

    def loss_function(self, recon_x, x, mu_, log_var_):
        bce = F.mse_loss(recon_x, x, reduction='sum')
        kl = -0.5 * torch.sum(
            1 + log_var_ - torch.square(mu_) - torch.exp(log_var_)
        )
        return bce + kl


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
            z_dim: int = 3,
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
            x = F.leaky_relu(x)
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
                x = F.leaky_relu(x)
            else:
                x = F.sigmoid(x)
        return x


def plot_latent(vae, data, num_batch=128):
    plt.figure()
    for i, (x, y) in enumerate(data):
        re_x, z, _, _ = vae(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batch:
            plt.colorbar()
            break
    plt.show()


def plot_reconstructed(vae, r0=(-5, 10), r1=(-10, 5), n=12):
    plt.figure()
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            re_x = vae.decoder(z)
            re_x = re_x.reshape(w, w).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = re_x
    plt.imshow(img, extent=[*r0, *r1])
    plt.show()


if __name__ == '__main__':
    print('Device: ', device)

    # load dataloader
    print('Loading DataLoader . . .')
    train_dataloader, test_dataloader = create_mnist_dataloader(n_batch=128)
    print('Completed Loading DataLoader\n')

    # create model and move to 'device'
    # model = VariationalAutoencoder(
    #     input_dim=(1, 28, 28),
    #     encoder_conv_channels=[32, 64, 64, 64],
    #     encoder_conv_kernel_size=[3, 3, 3, 3],
    #     encoder_conv_strides=[1, 2, 2, 1],
    #     decoder_conv_t_channels=[64, 64, 32, 1],
    #     decoder_conv_t_kernel_size=[3, 3, 3, 3],
    #     decoder_conv_t_strides=[1, 2, 2, 1],
    #     z_dim=16
    # )
    model = SimpleVAE(
        input_dim=(1, 28, 28),
        z_dim=2
    )
    model.to(device=device)
    summary(
        model,
        input_size=(256, 1, 28, 28),
        col_names=[
            'input_size',
            'output_size',
            'num_params',
            'kernel_size',
            'mult_adds'
        ],
    )

    # loss function
    # loss_fn = RMSELoss()

    # learning rate
    # lr = 1e-5

    # optimizer
    optimizer = optim.Adam(model.parameters())

    # training setting
    n_epochs = 1000
    step = 1

    train_losses = []
    test_losses = []
    early_stop_counter = 0

    print('\n====== Start training and test ======')
    start_time = datetime.now()
    print('Start time: ', start_time)
    for epoch in range(1, n_epochs + 1):
        # training
        model.train()
        loss_train = 0.0
        for i, (images, _) in enumerate(train_dataloader):
            images = images.to(device)
            batch_size = images.shape[0]
            reconstructed_images, _, mu, log_var = model(images)
            loss = model.loss_function(
                reconstructed_images, images, mu, log_var
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            if i == 0:  # for the first batch of each epoch
                plot_comparison(images, reconstructed_images, epoch)

        if epoch % step == 0:
            train_losses.append(loss_train / len(train_dataloader))

        # test
        model.eval()
        loss_test = 0.0

        with torch.no_grad():
            for images, _ in test_dataloader:
                images = images.to(device)
                batch_size = images.shape[0]
                reconstructed_images, _, mu, log_var = model(images)
                loss = model.loss_function(
                    reconstructed_images, images, mu, log_var
                )
                loss_test += loss.item()

            if epoch % step == 0:
                test_losses.append(loss_test / len(test_dataloader))

        if epoch % step == 0 or epoch == 1:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(
                'Time: {}, Epoch: {}, Training loss: {}, Test loss: {}'.format(
                    current_time,
                    epoch,
                    float(train_losses[-1]),
                    float(test_losses[-1])
                )
            )

        if min(test_losses) < test_losses[-1]:
            early_stop_counter += 1
        else:
            early_stop_counter = 0

        if early_stop_counter >= 5:
            print('early stopping')
            break

    end_time = datetime.now()
    print('training time: ', end_time - start_time)

    # plot
    train_loss = np.array(train_losses)
    test_loss = np.array(test_losses)

    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111)
    axis.plot(
        [i * step for i in range(1, len(train_losses) + 1)],
        train_loss, label='Train loss'
    )
    axis.plot(
        [i * step for i in range(1, len(test_losses) + 1)],
        test_loss, label='Test loss'
    )
    axis.set_title('Autoencoder')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Loss')
    axis.legend()

    save_path_as_pdf = './result/mnist_autoencoder.pdf'
    save_path_as_png = './result/mnist_autoencoder.png'
    plt.savefig(save_path_as_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_path_as_png, format='PNG', bbox_inches='tight')
    plt.show()

    plot_latent(model, test_dataloader)
    plot_reconstructed(model)
