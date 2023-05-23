import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets


def create_mnist_dataloader():
    # setting
    data_path = './datasets/MNIST_data'

    # load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, )),
         ]
    )
    train_set = datasets.MNIST(
        data_path, download=True, train=True, transform=transform
    )
    test_set = datasets.MNIST(
        data_path, download=True, train=False, transform=transform
    )

    # create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=True
    )
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
        return out


if __name__ == '__main__':
    # setting device
    device = torch.device('cuda') \
        if torch.cuda.is_available() else torch.device('cpu')

    # load dataloader
    train_dataloader, test_dataloader = create_mnist_dataloader()

    # create model and move to 'device'
    model = ConvImageClassifier(1, 28, 10, n_conv_channels=32)
    model.to(device=device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    learning_rate = 1e-3

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    n_epochs = 10000

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(n_epochs):
        # training
        loss = None
        for images, targets in train_dataloader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.shape[0]
            outputs = model(images)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            train_losses.append(loss.item())

        # test
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
                total += targets.shape[0]
                correct += int((predicted == targets).sum())

            if epoch % 100 == 0:
                test_losses.append(t_loss.item())
                test_accuracies.append(correct / total)

        if epoch % 100 == 0:
            print('Epoch: %d, Loss: %f, Test accuracy: %f'
                  % (epoch, float(loss), correct / total))

    # plot
    train_loss = np.array(train_losses)
    test_loss = np.array(test_losses)
    test_accuracies = np.array(test_accuracies)

    figure, axs = plt.subplots(1, 2)
    axs[0].plot([i * 100 for i in range(n_epochs // 100)], train_loss, label='Train_loss')
    axs[0].plot([i * 100 for i in range(n_epochs // 100)], test_loss, label='Test_loss')
    axs[0].set_title('Losses')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(range(n_epochs // 100), test_accuracies, label='Test_accuracies')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')

    figure.suptitle('MNIST Dataset Classifier')

    plt.show()
