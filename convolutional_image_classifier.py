from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


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


class ConvImageClassifier(nn.Module):
    def __init__(
            self, n_image_channels: int, image_size: int,
            n_class: int, n_conv_channels=32) -> None:
        super().__init__()
        self.n_image_channels = n_image_channels
        self.image_size = image_size
        self.n_class = n_class
        self.n_conv_channels = n_conv_channels
        self.conv1 = nn.Conv2d(
            n_image_channels, n_conv_channels,
            kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            n_conv_channels, n_conv_channels // 2,
            kernel_size=3, padding=1
        )
        self.linear1 = nn.Linear(
            n_conv_channels // 2 * image_size // 4 * image_size // 4,
            64
        )
        self.linear2 = nn.Linear(
            64,
            n_class
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.max_pool2d(F.relu(self.conv1(x)), 2)
        out = F.max_pool2d(F.relu(self.conv2(out)), 2)
        out = out.view(
            -1,
            (self.n_conv_channels // 2)
            * (self.image_size // 4) * (self.image_size // 4)
        )
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    # setting device
    device = torch.device('cuda') \
        if torch.cuda.is_available() else torch.device('cpu')
    print('device: ', device)

    # load dataloader
    print('Loading DataLoader . . .')
    train_dataloader, test_dataloader = create_mnist_dataloader()
    print('Completed Loading DataLoader\n')

    # create model and move to 'device'
    model = ConvImageClassifier(1, 28, 10, n_conv_channels=32)
    model.to(device=device)

    # loss function
    loss_fn = nn.NLLLoss()

    learning_rate = 1e-2

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters())

    n_epochs = 1000
    step = 1

    train_losses = []
    test_losses = []
    test_accuracies = []

    print('====== Start training and test ======')
    for epoch in range(1, n_epochs + 1):
        # training
        model.train()
        loss_train = 0.0
        for images, targets in train_dataloader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.shape[0]
            outputs = model(images)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch % step == 0:
            train_losses.append(loss_train / len(train_dataloader))

        # test
        model.eval()
        loss_test = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in test_dataloader:
                images = images.to(device)
                targets = targets.to(device)
                batch_size = images.shape[0]
                outputs = model(images)
                t_loss = loss_fn(outputs, targets)
                _, predicted = torch.max(outputs, dim=1)
                loss_test += t_loss.item()
                total += targets.shape[0]
                correct += int((predicted == targets).sum())

            if epoch % step == 0:
                test_losses.append(loss_test / len(test_dataloader))
                test_accuracies.append(correct / total)

        if epoch % step == 0 or epoch == 1:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(
                'Log: {}, Epoch: {}, Training loss: {}, Test loss: {},'
                ' Test accuracy: {}'.format(
                    current_time, epoch,
                    float(loss_train / len(train_dataloader)),
                    float(loss_test / len(test_dataloader)),
                    correct / total
                )
            )

    # plot
    train_loss = np.array(train_losses)
    test_loss = np.array(test_losses)
    test_accuracies = np.array(test_accuracies)

    figure, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].plot(
        [i * step for i in range(1, n_epochs // step + 1)],
        train_loss, label='Train_loss'
    )
    axs[0].plot(
        [i * step for i in range(1, n_epochs // step + 1)],
        test_loss, label='Test_loss'
    )
    axs[0].set_title('Losses')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(range(n_epochs // step), test_accuracies,
                label='Test_accuracies')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')

    figure.suptitle('MNIST Dataset Classifier')

    save_path_as_pdf = './result/mnist_conv_nn_adam.pdf'
    save_path_as_png = './result/mnist_conv_nn_adam.png'
    plt.savefig(save_path_as_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_path_as_png, format='png', bbox_inches='tight')
    plt.show()
