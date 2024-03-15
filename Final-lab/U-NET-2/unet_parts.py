import torch
import torch.nn as nn
import torch.nn.functional as F

'''
连续的两次卷积操作，每次卷积后都跟着一个批量归一化（Batch Normalization）和ReLU激活函数。
用于特征提取，对应unet网络中频繁的连续两次卷积
'''
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 组合卷积层、归一化和激活函数，并封装成一个顺序执行的容器
        self.double_conv = nn.Sequential(
            # 卷积，输入1通道， 输出6通道，图像大小不变
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 批量归一化，用于增加训练速度和模型稳定性
            nn.BatchNorm2d(out_channels),
            # 激活函数，引入非线性，inplace=True指定在原地修改数据，减少内存占用
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            # 使用双线性插值进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 使用转置卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is [C, H, W]
        diff_y = torch.tensor([x2.size()[2] - x1.size()[2]])
        diff_x = torch.tensor([x2.size()[3] - x1.size()[3]])

        # 因为要将x1和x2在通道维度拼接，所以要保证大小一致
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)