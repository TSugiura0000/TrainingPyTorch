from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset


def create_iris_dataloader(batch_size=32):
    # load iris dataset from scikit-learn
    iris = load_iris()
    X_train, X_test, y_train, y_test = \
        train_test_split(iris.data, iris.target, test_size=0.3, random_state=11)

    # convert to torch tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    # create dataset
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # create dataloader
    train_dataloader_ = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader_ = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )
    return train_dataloader_, test_dataloader_


class NN4Iris(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 3)

    def forward(self, x):
        out = x.view(-1, 4)
        out = torch.relu(self.linear1(out))
        out = torch.relu(self.linear2(out))
        out = torch.relu(self.linear3(out))
        out = torch.relu(self.linear4(out))
        out = torch.log_softmax(out, dim=1)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader, test_dataloader = create_iris_dataloader(batch_size=32)
    model = NN4Iris()
    model.to(device=device)
    loss_fn = nn.NLLLoss()
    learning_rate = 1e-3
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    n_epochs = 1000
    step = 10
    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, n_epochs + 1):
        # training
        model.train()
        loss_train = 0.0
        for data, targets in train_dataloader:
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
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
            for data, targets in test_dataloader:
                data = data.to(device)
                targets = targets.to(device)
                outputs = model(data)
                t_loss = loss_fn(outputs, targets)
                _, predicted = torch.max(outputs, dim=1)
                loss_test += t_loss.item()
                total += targets.shape[0]
                correct += int((predicted == targets).sum())

            if epoch % step == 0:
                test_losses.append(t_loss.item())
                test_accuracies.append(correct / total)

        if epoch == 1 or epoch % step == 0:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%H:%S')
            print(
                'Log: {}, Epoch: {}, Training Loss: {}, Test Loss: {} '
                'Test accuracy: {}'.format(
                    current_time, epoch,
                    float(loss_train / len(train_dataloader)),
                    float(loss_test / len(test_dataloader)),
                    correct / total
                ))

    # plot
    train_loss = np.array(train_losses)
    test_loss = np.array(test_losses)
    test_accuracies = np.array(test_accuracies)

    figure, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].plot([i * step for i in range(1, n_epochs // step + 1)], train_loss, label='Train_loss')
    axs[0].plot([i * step for i in range(1, n_epochs // step + 1)], test_loss, label='Test_loss')
    axs[0].set_title('Losses')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot([i * step for i in range(1, n_epochs // step + 1)], test_accuracies, label='Test_accuracies')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')

    figure.suptitle('Iris Dataset Classifier')

    save_path_as_pdf = f'./result/iris_nn_sgd.pdf'
    plt.savefig(save_path_as_pdf, format='pdf', bbox_inches='tight')

    plt.show()
