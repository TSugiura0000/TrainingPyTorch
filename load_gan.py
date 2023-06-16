import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from gan import Discriminator, Generator, create_gif


# setting device
device = torch.device('cuda') \
    if torch.cuda.is_available() else torch.device('cpu')


def create_morphing_image(generator: nn.Module, z_: torch.Tensor,
                          save_path_: None or str, show: bool = True,
                          gif_name: str = None, fps: int = 5) -> None:
    if save_path_ is None:
        dir_path = './result/gan_generated_images'
    else:
        dir_path = save_path_

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    z_ = z_.to(device=device)
    generator.to(device=device)
    generator.eval()
    with torch.no_grad():
        image = generator(z_)
    generator.train()
    image = image.cpu().detach().numpy()

    for i in tqdm(range(len(image))):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        plt.imshow(image[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        # plt.savefig(os.path.join(save_path_, f'4morph_{i}.pdf'))
        plt.savefig(os.path.join(save_path_, f'4morph_{i}.png'))
        if show:
            plt.show()
        else:
            plt.close()

    create_gif(gif_name, fps)


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

    # morphing
    z_dim = 100
    n_batch = 256
    z1 = torch.randn(n_batch, z_dim)
    z2 = torch.randn(n_batch, z_dim)
    line_space = torch.linspace(0, 1, steps=n_batch).unsqueeze(1)
    z = line_space * z1 + (1 - line_space) * z2

    save_path = './result/GIF/for_gif'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    create_morphing_image(g, z, save_path, show=False,
                          gif_name='morphing_fps5.gif', fps=5)

