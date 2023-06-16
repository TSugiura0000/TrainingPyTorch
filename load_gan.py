import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn

from gan import Discriminator, Generator


# setting device
device = torch.device('cuda') \
    if torch.cuda.is_available() else torch.device('cpu')


def plot_generated_image(generator: nn.Module, z_: torch.Tensor,
                         save_path: None or str, show: bool = True) -> None:
    if save_path is None:
        dir_path = './result/gan_generated_images'
    else:
        dir_path = save_path

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    z_.to(device=device)
    timestamp = datetime.now()
    image = generator(z_)
    image = image.cpu().detach().numpy()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.imshow(image[0].reshape(28, 28), cmap='gray')
    ax.axis('off')
    plt.savefig(
        f'./result/gan_generated_images/z2x_{timestamp}.pdf'
    )
    if show:
        plt.show()


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

    # plot generated image
    z = torch.randn(256, 100)
    plot_generated_image(g, z, show=False)
