from torch import nn
import torch
import torch.nn.functional as F

# ディープ・ニューラルネットワークの構築
class Net(nn.Module):
    def __init__(self, n_in, n_mid=64, n_out=36):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(in_features=32 * 19 * 19, out_features=n_mid)
        self.fc2 = nn.Linear(in_features=n_mid, out_features=n_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 32 * 19 * 19)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
