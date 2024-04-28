import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, train_config):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = getattr(nn, train_config['activation_function'])()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x + residual


class ResidualCNN(nn.Module):
    def __init__(self, train_config):
        super(ResidualCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.act1 = getattr(nn, train_config['activation_function'])()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1 = ResidualBlock(32, train_config)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act2 = getattr(nn, train_config['activation_function'])()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res2 = ResidualBlock(64, train_config)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.act3 = getattr(nn, train_config['activation_function'])()
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.res1(x)
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.res2(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x
