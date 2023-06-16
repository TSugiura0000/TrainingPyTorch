import torch
import torch.nn as nn
from torchviz import make_dot

from gan import Discriminator, Generator

if __name__ == '__main__':
    n_batch = 256
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
    z = torch.randn(n_batch, 100)  # ここでは適当な入力データを作成しています
    x = torch.randn(n_batch, 1, 28, 28)
    y = torch.randn(n_batch, 1)

    # モデルのforwardパスを実行します
    fake_x = g(z)
    p, _ = d(x, y)

    # 計算グラフを視覚化します
    g_dot = make_dot(fake_x, params=dict(g.named_parameters()))
    g_dot.view()

    # d_dot = make_dot(p, params=dict(d.named_parameters()))
    # d_dot.view()


