import torch
import torch.nn as nn
import torchvision.transforms as transforms


class SimpleAlexNet(nn.Module):
    def __init__(self, train_config, num_classes=100):
        super(SimpleAlexNet, self).__init__()
        self.resize = transforms.Resize((244, 244))
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            getattr(nn, train_config['activation_function'])(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            getattr(nn, train_config['activation_function'])(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            getattr(nn, train_config['activation_function'])(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            getattr(nn, train_config['activation_function'])(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            getattr(nn, train_config['activation_function'])(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.resize(x)  # Resize the input image
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
