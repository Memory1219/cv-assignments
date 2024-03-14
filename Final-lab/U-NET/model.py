import torch
import torch.nn as nn
import torch.nn.functional as F


def crop_tensor(target_tensor, tensor_to_crop):
    target_size = target_tensor.size()[2:]  # Get the size of the target tensor
    tensor_size = tensor_to_crop.size()[2:]  # Get the size of the tensor to crop
    delta = [tensor_size[i] - target_size[i] for i in range(2)]
    crop_start = [delta[i] // 2 for i in range(2)]
    crop_end = [crop_start[i] + target_size[i] for i in range(2)]
    return tensor_to_crop[:, :, crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]]


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)

        x = self.up1(x3)
        x2_cropped = crop_tensor(x, x2)
        x = torch.cat([x, x2_cropped], dim=1)
        x = self.conv3(x)
        x = self.up2(x)
        x1_cropped = crop_tensor(x, x1)
        x = torch.cat([x, x1_cropped], dim=1)
        x = self.conv4(x)
        logits = self.outc(x)
        return logits


