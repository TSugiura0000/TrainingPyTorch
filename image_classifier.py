import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = './datasets/MNIST_data/'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

train_set = datasets.MNIST(data_path, download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

test_set = datasets.MNIST(data_path, download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
)
model.to(device)

loss_fn = nn.NLLLoss()

learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

n_epochs = 100


if __name__ == '__main__':
    train_losses = []
    test_losses = []
    test_accuracies = []

    # training
    for epoch in range(n_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]
            outputs = model(images.view(batch_size, -1))
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        # test
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                batch_size = images.shape[0]
                outputs = model(images.view(batch_size, -1))
                t_loss = loss_fn(outputs, labels)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

            test_losses.append(t_loss.item())
            test_accuracies.append(correct / total)

        if epoch % 1 == 0:
            print('Epoch: %d, Loss: %f, Test accuracy: %f' % (epoch, float(loss), correct / total))

    # plot
    train_loss = np.array(train_losses)
    test_loss = np.array(test_losses)
    test_accuracies = np.array(test_accuracies)

    figure, axs = plt.subplots(1, 2)
    axs[0].plot(range(n_epochs), train_loss, label='Train_loss')
    axs[0].plot(range(n_epochs), test_loss, label='Test_loss')
    axs[0].set_title('Losses')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(range(n_epochs), test_accuracies, label='Test_accuracies')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')

    figure.suptitle('MNIST Dataset Classifier')

    plt.show()
