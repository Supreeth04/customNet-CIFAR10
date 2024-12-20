import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, dilation=dilation
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # C1 Block - Initial features extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )
        
        # C2 Block
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            DepthwiseSeparableConv(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )
        
        # C3 Block
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(64, 96, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1),
            DepthwiseSeparableConv(96, 96, kernel_size=3),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1),
        )
        
        # C4 Block
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv(96, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            DepthwiseSeparableConv(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)
        
        # Add dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x