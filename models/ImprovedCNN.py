from torch import nn


class ImprovedCNN(nn.Module):
    def __init__(self, train_config):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization after conv1
        self.act1 = getattr(nn, train_config['activation_function'])()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization after conv2
        self.act2 = getattr(nn, train_config['activation_function'])()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch Normalization after conv3
        self.act3 = getattr(nn, train_config['activation_function'])()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  # Batch Normalization after conv4
        self.act4 = getattr(nn, train_config['activation_function'])()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn5 = nn.BatchNorm1d(512)  # Batch Normalization after fc1
        self.act5 = getattr(nn, train_config['activation_function'])()
        self.dropout = nn.Dropout(train_config['dropout_rate'])
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(self.act5(self.bn5(self.fc1(x))))
        x = self.fc2(x)
        return x
