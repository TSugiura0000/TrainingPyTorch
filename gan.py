import os

import imageio.v2 as imageio
import numpy as np
import psutil
import GPUtil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary

# setting device
device = torch.device('cuda') \
    if torch.cuda.is_available() else torch.device('cpu')

# create directory for generated images
gen_image_path = 'result/gan_generated_images'
if not os.path.exists(gen_image_path):
    os.makedirs(gen_image_path)


def print_system_info():
    # CPU usage
    cpu_usage = psutil.cpu_percent()
    print(f"CPU usage: {cpu_usage}%")

    # Memory usage
    memory_info = psutil.virtual_memory()
    memory_usage = (memory_info.used / memory_info.total) * 100
    print(f"Memory usage: {memory_usage}%")

    # GPU usage
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.load*100}%")


def create_mnist_dataloader(n_batch: int = 64):
    # setting
    data_path = './datasets/MNIST_data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # load dataset
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    train_set = datasets.MNIST(
        data_path, download=True, transform=transform
    )

    # create dataloader
    train_loader = DataLoader(train_set, batch_size=n_batch, shuffle=True)
    return train_loader


def plot_comparison(real_image: torch.Tensor, fake_image: torch.Tensor,
                    epoch: int, image_num: int = 10) -> None:
    dir_path = 'result/gan_comparison_images/'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    real_image = real_image.cpu().numpy()
    fake_image = fake_image.cpu().detach().numpy()

    fig = plt.figure(figsize=(20, 4))
    gs = gridspec.GridSpec(2, image_num)
    gs.update(wspace=0.05, hspace=0.05)

    for i in range(image_num):
        ax = plt.subplot(gs[0, i])
        plt.imshow(real_image[i].reshape(28, 28), cmap='gray')
        ax.axis('off')

        ax = plt.subplot(gs[1, i])
        plt.imshow(fake_image[i].reshape(28, 28), cmap='gray')
        ax.axis('off')

    plt.savefig(
        f'./result/gan_comparison_images/epoch_{epoch}.pdf', bbox_inches='tight'
    )
    plt.close(fig)


def create_gif() -> None:
    # setting
    dir_path = './result/gan_generated_images'
    gif_name = './result/generated_images/GIF.gif'

    # sort images
    image_files = os.listdir(dir_path)
    image_files.sort()

    # load images
    images = []
    for filename in sorted(image_files):
        if filename.endswith('.png'):
            file_path = os.path.join(dir_path, filename)
            images.append(imageio.imread(file_path))

    # create GIF
    imageio.mimsave(gif_name, images, duration=0.2)


def plot_loss(d_real: list, d_fake: list, d_mean: list, g: list) -> None:
    d_real_loss = np.array(d_real)
    d_fake_loss = np.array(d_fake)
    d_mean_loss = np.array(d_mean)
    g_loss = np.array(g)

    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111)
    axis.plot(
        np.arange(1, len(d_real_loss) + 1), d_real_loss,
        label='Discriminator Loss (Real data)'
    )
    axis.plot(
        np.arange(1, len(d_fake_loss) + 1), d_fake_loss,
        label='Discriminator Loss (Fake data)'
    )
    axis.plot(
        np.arange(1, len(d_mean_loss) + 1), d_mean_loss,
        label='Discriminator Loss (Mean)'
    )
    axis.plot(
        np.arange(1, len(g_loss) + 1), g_loss, label='Generator Loss'
    )
    axis.set_title('Generative Adversarial Networks')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Loss')
    axis.legend()

    save_path_as_pdf = 'result/loss/mnist_gan.pdf'
    save_path_as_png = 'result/loss/mnist_gan.png'
    plt.savefig(save_path_as_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_path_as_png, format='PNG', bbox_inches='tight')
    # plt.show()


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
        (conv_kernel_size, conv_kernel_filter) must be same.
        """
        super().__init__()
        dropout_layer = nn.Dropout2d(p=dropout_rate)
        activate_fn = nn.ReLU()
        num_pooling = len(conv_kernel_filter) - 1
        if conv_kernel_size is None:
            conv_kernel_size = [3, 3, 3]
        if conv_kernel_filter is None:
            conv_kernel_filter = [64, 64, 128]
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
                padding=k_size // 2
            )
                for in_chan, out_chan, k_size
                in zip(
                [self._input_dim[0]] + self._conv_kernel_filter[0:-1],
                self._conv_kernel_filter,
                self._conv_kernel_size
            )]
        )
        self._last_linear = nn.Linear(
            ((self._input_dim[1] // (2 ** num_pooling)) ** 2)
            * conv_kernel_filter[-1],
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
        (conv_kernel_size, conv_kernel_filter) must be same.
        """
        super().__init__()
        batch_norm_layers = []
        dropout_layer = nn.Dropout2d(p=dropout_rate)
        activate_fn = nn.ReLU()
        init_linear_features = \
            init_linear_size[0] * init_linear_size[1] * init_linear_size[2]
        if conv_kernel_size is None:
            conv_kernel_size = [3, 3, 3]
        if conv_kernel_filter is None:
            conv_kernel_filter = [128, 64, 1]
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
                padding=k_size // 2
            )
                for in_chan, out_chan, k_size
                in zip(
                [self._init_linear_size[0]] + self._conv_kernel_filter[0:-1],
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
            if i != len(self._nn_batch_norm_layers) - 1:
                x = self._nn_batch_norm_layers[i + 1](x)
                x = self._activate_fn(x)
            else:
                x = torch.sigmoid(x)
        return x


def train_gan(
        epoch_num_: int,
        data_loader: DataLoader,
        discriminator_: nn.Module,
        generator_: nn.Module,
        d_optimizer_: optim,
        g_optimizer_: optim,
        z_dim_,
        batch_size_
):
    d_real_losses = []
    d_fake_losses = []
    d_mean_losses = []
    g_losses = []
    fixed_z = torch.randn(batch_size_, z_dim_, device=device)
    for epoch in tqdm(range(epoch_num_)):
        d_real_l = 0
        d_fake_l = 0
        d_mean_l = 0
        g_l = 0
        for x, _ in data_loader:
            x = x.to(device=device)
            batch_size_ = x.size(0)

            # create labels
            real_label = torch.ones((batch_size_, 1), device=device)
            fake_label = torch.zeros((batch_size_, 1), device=device)

            # --- train discriminator --- #
            d_optimizer_.zero_grad()
            # real data
            _, d_real_loss = discriminator_(x, real_label)
            d_real_l += d_real_loss.item()
            # fake data
            z = torch.randn(batch_size_, z_dim_, device=device)
            fake_x = generator_(z)
            _, d_fake_loss = discriminator_(fake_x.detach(), fake_label)
            d_fake_l += d_fake_loss.item()
            d_loss = d_real_loss + d_fake_loss
            d_mean_l += d_loss.item() / 2

            d_loss.backward()
            d_optimizer_.step()
            # --- train discriminator --- #

            # --- train generator --- #
            g_optimizer_.zero_grad()
            _, g_loss = discriminator_(fake_x, real_label)
            g_l += g_loss.item()

            g_loss.backward()
            g_optimizer_.step()
            # --- train generator --- #

            # save figure
            generator.eval()
            with torch.no_grad():
                fake_images = generator(fixed_z)
            generator.train()

            grid = torchvision.utils.make_grid(
                fake_images[:64], padding=2, normalize=True
            )
            torchvision.utils.save_image(
                grid, f'result/gan_generated_images/epoch_{epoch + 1}.pdf')
            torchvision.utils.save_image(
                grid, f'result/gan_generated_images/epoch_{epoch + 1}.png')
            plot_comparison(x, fake_x, epoch + 1, image_num=10)

        d_real_losses.append(d_real_l / len(data_loader))
        d_fake_losses.append(d_fake_l / len(data_loader))
        d_mean_losses.append(d_mean_l / len(data_loader))
        g_losses.append(g_l / len(data_loader))

        print(f'Epoch: {epoch + 1}/{epoch_num} \t '
              f'Discriminator Loss: (Real: {d_real_losses[epoch]:.4f}, '
              f'Fake: {d_fake_losses[epoch]:.4f}, '
              f'Mean: {d_mean_losses[epoch]:.4f})\t'
              f'Generator Loss: {g_losses[epoch]:.4f}')

    plot_loss(d_real_losses, d_fake_losses, d_mean_losses, g_losses)


if __name__ == '__main__':
    # device
    print('Device: ', device)

    # load dataloader
    print('Loading DataLoader . . .')
    batch_size = 256
    train_dataloader = create_mnist_dataloader(n_batch=batch_size)
    print('Completed Loading DataLoader\n')

    # setting model
    z_dim = 100
    discriminator = Discriminator(
        input_dim=(1, 28, 28),
        conv_kernel_size=[5, 5, 5, 5],
        conv_kernel_filter=[64, 64, 128, 128],
        dropout_rate=0.4,
        activation='relu'
    )
    generator = Generator(
        z_dim=z_dim,
        init_linear_size=(64, 7, 7),
        conv_kernel_size=[5, 5, 5, 5],
        conv_kernel_filter=[128, 64, 64, 1],
        dropout_rate=0.0,
        batch_norm=True,
        activation='relu',
        output_size=(1, 28, 28)
    )
    discriminator.to(device=device)
    generator.to(device=device)

    # summary
    img_data = torch.randn(batch_size, 1, 28, 28, device=device)
    labels = torch.randn(batch_size, 1, device=device)
    summary(
        discriminator,
        input_data=[img_data, labels],
        col_names=[
            'input_size',
            'output_size',
            'num_params',
            'kernel_size',
            'mult_adds'
        ],
        batch_dim=0
    )
    summary(
        generator,
        input_size=(batch_size, z_dim),
        col_names=[
            'input_size',
            'output_size',
            'num_params',
            'kernel_size',
            'mult_adds'
        ]
    )

    # setting optimizer
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0008)
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0004)

    # training gan
    epoch_num = 100
    train_gan(
        epoch_num_=epoch_num,
        data_loader=train_dataloader,
        discriminator_=discriminator,
        generator_=generator,
        d_optimizer_=discriminator_optimizer,
        g_optimizer_=generator_optimizer,
        z_dim_=z_dim,
        batch_size_=batch_size
    )

    # save models' parameters
    model_dir = './model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(discriminator.state_dict(),
               os.path.join(model_dir, 'discriminator.pth'))
    torch.save(generator.state_dict(),
               os.path.join(model_dir, 'generator.pth'))

    # create GIF image
    create_gif()
