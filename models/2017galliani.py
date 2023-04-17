import torch
import torch.nn as nn
from math import sqrt

class DenseU(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DenseU, self).__init__()
        self.convinput = self.convlayer(in_channel, 64, 3)
        self.dd1 = Denseblock_Down(64)
        self.down=nn.MaxPool2d(2)
        self.dd2 = Denseblock_Down(64*2)
        self.dd3 = Denseblock_Down(64*3)
        self.dd4 = Denseblock_Down(64*4)
        self.dd5 = Denseblock_Down(64*5)
        self.db = Denseblock(64*6)
        self.du1 = Denseblock_Up(64,64)
        self.du2 = Denseblock_Up(64*2,64)
        self.du3 = Denseblock_Up(64*2,64)
        self.du4 = Denseblock_Up(64*2,64)
        self.du5=Denseblock_Up(64*2,out_channel)


    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        Lrelu = nn.LeakyReLU(0.2, inplace=True)
        layers = filter(lambda x: x is not None, [pader, conver, Lrelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        input1=self.convinput(x)
        down1=self.dd1(input1)
        input1_down=self.down(input1)
        down1_c=torch.cat([input1_down,down1],1)
        down2 = self.dd2(down1_c)
        down1_c_down = self.down(down1_c)
        down2_c = torch.cat([down1_c_down, down2], 1)
        down3 = self.dd3(down2_c)
        down2_c_down = self.down(down2_c)
        down3_c = torch.cat([down2_c_down, down3], 1)
        down4 = self.dd4(down3_c)
        down3_c_down = self.down(down3_c)
        down4_c = torch.cat([down3_c_down, down4], 1)
        down5 = self.dd5(down4_c)
        down4_c_down = self.down(down4_c)
        down5_c = torch.cat([down4_c_down, down5], 1)
        centrel=self.db(down5_c)
        up1 = self.du1(centrel)
        up1_c=torch.cat([down4, up1], 1)
        up2 = self.du2(up1_c)
        up2_c = torch.cat([down3, up2], 1)
        up3 = self.du3(up2_c)
        up3_c = torch.cat([down2, up3], 1)
        up4 = self.du4(up3_c)
        up4_c = torch.cat([down1, up4], 1)
        up5 = self.du5(up4_c)
        return up5

class Denseblock(nn.Module):
    def __init__(self, in_channels):
        super(Denseblock, self).__init__()
        self.conv1 = self.convlayer(in_channels, 16, 3)
        self.conv2 = self.convlayer(16 + in_channels, 16, 3)
        self.conv3 = self.convlayer(32 + in_channels, 16, 3)
        self.conv4 = self.convlayer(48 + in_channels, 16, 3)
        self.conv = self.convlayer(64, 64, 1)

    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        Lrelu = nn.LeakyReLU(0.2, inplace=True)
        layers = filter(lambda x: x is not None, [pader, conver, Lrelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out1_c=torch.cat([x, out1], 1)
        out2 = self.conv2(out1_c)
        out2_c = torch.cat([out1_c, out2], 1)
        out3=self.conv3(out2_c)
        out3_c = torch.cat([out2_c, out3], 1)
        out4 = self.conv4(out3_c)
        out4 = torch.cat([out1,out2,out3, out4], 1)
        output =self.conv(out4)
        return output

class Denseblock_Down(nn.Module):
    def __init__(self, in_channels):
        super(Denseblock_Down, self).__init__()
        self.conv1 = self.convlayer(in_channels, 16, 3)
        self.conv2 = self.convlayer(16 + in_channels, 16, 3)
        self.conv3 = self.convlayer(32 + in_channels, 16, 3)
        self.conv4 = self.convlayer(48 + in_channels, 16, 3)
        self.conv = self.convlayer(64, 64, 1)
        self.pooling=nn.MaxPool2d(2)

    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        Lrelu = nn.LeakyReLU(0.2, inplace=True)
        layers = filter(lambda x: x is not None, [pader, conver, Lrelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out1_c=torch.cat([x, out1], 1)
        out2 = self.conv2(out1_c)
        out2_c = torch.cat([out1_c, out2], 1)
        out3=self.conv3(out2_c)
        out3_c = torch.cat([out2_c, out3], 1)
        out4 = self.conv4(out3_c)
        out4 = torch.cat([out1,out2,out3, out4], 1)
        in_pooling=self.conv(out4)
        output = self.pooling(in_pooling)
        return output

class Denseblock_Up(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Denseblock_Up, self).__init__()
        self.conv1 = self.convlayer(in_channels, 16, 3)
        self.conv2 = self.convlayer(16 + in_channels, 16, 3)
        self.conv3 = self.convlayer(32 + in_channels, 16, 3)
        self.conv4 = self.convlayer(48 + in_channels, 16, 3)
        self.conv = self.convlayer(64, out_channels*4, 1)
        self.conv_subpixel =nn.PixelShuffle(2)

    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        Lrelu = nn.LeakyReLU(0.2, inplace=True)
        layers = filter(lambda x: x is not None, [pader, conver, Lrelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out1_c=torch.cat([x, out1], 1)
        out2 = self.conv2(out1_c)
        out2_c = torch.cat([out1_c, out2], 1)
        out3=self.conv3(out2_c)
        out3_c = torch.cat([out2_c, out3], 1)
        out4 = self.conv4(out3_c)
        out4 = torch.cat([out1,out2,out3, out4], 1)
        in_subpixel=self.conv(out4)
        output = self.conv_subpixel(in_subpixel)
        return output