import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionLikeCNN(nn.Module):
    def __init__(self, train_config):
        super(InceptionLikeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.act1 = getattr(nn, train_config['activation_function'])()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Inception-like module
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=1)
        self.conv2_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_5 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.act2 = getattr(nn, train_config['activation_function'])()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(192 * 8 * 8, 512) # Updated size due to concatenation
        self.act3 = getattr(nn, train_config['activation_function'])()
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x1 = self.conv2_1(x)
        x3 = self.conv2_3(x)
        x5 = self.conv2_5(x)
        x = torch.cat([x1, x3, x5], 1) # Concatenate feature maps
        x = self.pool2(self.act2(x))
        x = x.view(-1, 192 * 8 * 8)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x
