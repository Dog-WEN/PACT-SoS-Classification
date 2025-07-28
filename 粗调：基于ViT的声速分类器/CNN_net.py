import torch
from torch import nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)

class VGGBlock(nn.Module):
    def __init__(self, num_conv, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(VGGBlock, self).__init__()
        layers = []
        for _ in range(num_conv):
            layers.append(ConvBlock(in_channels, out_channels, kernel_size, stride, padding))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(2))
        self.vgg = nn.Sequential(*layers)

    def forward(self, x):
        return self.vgg(x)

class NiNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(NiNBlock, self).__init__()
        self.nin = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.nin(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = None
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.bn3(self.conv3(X))
        Y += X
        return F.relu(Y)

class AlexNet(nn.Module):
    def __init__(self, in_channels=1):
        super(AlexNet, self).__init__()
        self.alex_net = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=13, stride=3, padding=0),
            nn.MaxPool2d(2),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(2),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            ConvBlock(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(384 * 8 * 8, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.alex_net(x)

class VGG(nn.Module):
    def __init__(self, in_channels=1):
        super(VGG, self).__init__()
        self.vgg = nn.Sequential(
            VGGBlock(num_conv=1, in_channels=in_channels, out_channels=64, padding=0),
            VGGBlock(num_conv=2, in_channels=64, out_channels=128, padding=0),
            VGGBlock(num_conv=1, in_channels=128, out_channels=256, padding=1),
            VGGBlock(num_conv=1, in_channels=256, out_channels=384, padding=1),
            VGGBlock(num_conv=1, in_channels=384, out_channels=384, padding=1),

            nn.Flatten(),
            nn.Linear(384 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.vgg(x)

class NiN(nn.Module):
    def __init__(self, in_channels=1):
        super(NiN, self).__init__()
        self.nin = nn.Sequential(
            NiNBlock(in_channels=in_channels, out_channels=64, kernel_size=7, stride=3, padding=0),
            nn.MaxPool2d(2),
            NiNBlock(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(2),
            NiNBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2),

            nn.Dropout(0.5),
            NiNBlock(in_channels=256, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.nin(x)

class ResNet(nn.Module):
    def __init__(self, in_channels=1):
        super(ResNet, self).__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=128, use_1x1conv=True, stride=2),
            ResidualBlock(in_channels=128, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=256, use_1x1conv=True, stride=2),
            ResidualBlock(in_channels=256, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=384, use_1x1conv=True, stride=2),
            ResidualBlock(in_channels=384, out_channels=384),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(384, 2)
        )

    def forward(self, x):
        return self.resnet(x)

# 测试
if __name__ == '__main__':
    test_img = torch.rand([16, 1, 250, 250])
    test_net = ResNet(in_channels=1)
    print(test_net(test_img).shape)