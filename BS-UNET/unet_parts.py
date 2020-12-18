import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.functional import relu6
from torch.nn import *


# 深度卷积块
class depthwise_block(nn.Module):
    def __init__(self, inplanes, outplanes, strides):
        super(depthwise_block, self).__init__()
        self.zeropad = ZeroPad2d(padding=1)
        self.DW = Conv2d(inplanes, inplanes,  # 深度卷积,输入和输出通道一致
                         kernel_size=3, stride=strides,
                         padding=0, groups=inplanes,  # groups=inplanes是实现深度卷积的重点
                         bias=False)
        self.BN_1 = BatchNorm2d(inplanes, momentum=0.1)
        self.BN_2 = BatchNorm2d(outplanes, momentum=0.1)
        self.conv = Conv2d(inplanes, outplanes, kernel_size=1, stride=1)
        # self.relu=ReLU()

    def forward(self, x):
        x = self.zeropad(x)
        x = self.DW(x)
        x = self.BN_1(x)
        # x=self.relu(x)
        x = relu6(x)
        x = self.conv(x)
        x = self.BN_2(x)
        # x=self.relu(x)
        x = relu6(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# in_channels和out_channels可以灵活设置

# Down模块
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# up模块
class up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
