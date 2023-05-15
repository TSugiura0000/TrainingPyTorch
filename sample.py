import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

num_epochs = 100
num_batch = 100
learning_rate = 0.001
image_size = 28 * 28


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    train_data = torchvision.datasets.FashionMNIST(
        './datasets', train=True, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=100, shuffle=True
    )

    # GPU mode
    device = torch.device('mps') \
        if torch.backends.mps.is_available() else torch.device('cpu')
    model = NeuralNet().to(device)
    model.train()

    # setting loss function
    criterion = nn.CrossEntropyLoss()
    # setting optimize method
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        loss_sum = 0

        for inputs, labels in train_dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            inputs = inputs.view(-1, image_size)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss_sum += loss

            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch+1}/{num_epochs},'
              f' Loss: {loss_sum / len(train_dataloader)}')
