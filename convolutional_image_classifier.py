import torch
from torch import nn
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset, DataLoader


def create_iris_dataloader(batch_size=32):
    # load iris dataset from scikit-learn
    iris = load_iris()

    # convert to torch tensor
    data = torch.tensor(iris['data'], dtype=torch.float32)
    targets = torch.tensor(iris['target'], dtype=torch.int64)

    # create dataset
    dataset = TensorDataset(data, targets)

    # return dataloader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class ConvImageClassifier(nn.Module):
    def __init__(self, n_image_channels, n_conv_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(
            n_image_channels, n_conv_channels,
            kernel_size=3, padding=1
        ),
        self.conv2 = nn.Conv2d(
            n_conv_channels, n_conv_channels // 2,
            kernel_size=3, padding=1
        ),
        self.linear1 = nn.Linear()


if __name__ == '__main__':
    pass
