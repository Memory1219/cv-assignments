from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # unet第一步，对输入图像进行两次卷积
        self.inc = DoubleConv(n_channels, 64)

        # 对应四次下采样，unet中下采样后的卷积、激活等过程也融合在Down中
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # 对应四次上采样，unet中上采样后的拼接、卷积等过程也融入Up函数中
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # 最后将输出的64通道通过卷积映射到n_classes个通道，对应每个像素属于每个类别的概率
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output


if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)
