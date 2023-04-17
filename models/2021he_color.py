import torch
import torch.nn as nn
from math import sqrt
from math import pi


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.gelu1 =nn.LeakyReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.gelu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.gelu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class HSRnet(nn.Module):
    def __init__(self,in_channels,out_channel):
        super(HSRnet, self).__init__()
        self.conv_B = self.convlayer(in_channels, 3, 3)
        self.conv_G = self.convlayer(in_channels, 3, 3)
        self.conv_R = self.convlayer(in_channels, 3, 3)
        self.convinputb = self.convlayer(3, 1, 1)
        self.convinputg = self.convlayer(3, 1, 1)
        self.convinputr = self.convlayer(3, 1, 1)
        self.conv1 = self.convlayer(out_channel, out_channel, 3)
        self.e1 = ChannelAttention(out_channel)
        self.HSI1 = HSI_block(out_channel, out_channel)
        self.en1 = ChannelAttention(out_channel)
        self.conv2 = self.convlayer(out_channel, out_channel, 3)
        self.e2 = ChannelAttention(out_channel)
        self.HSI2 = HSI_block(out_channel, out_channel)
        self.en2 = ChannelAttention(out_channel)
        self.conv3 = self.convlayer(out_channel, out_channel, 3)
        self.e3 = ChannelAttention(out_channel)
        self.HSI3 = HSI_block(out_channel, out_channel)
        self.en3 = ChannelAttention(out_channel)
        self.conv4 = self.convlayer(out_channel, out_channel, 3)
        self.e4 = ChannelAttention(out_channel)
        self.HSI4 = HSI_block(out_channel, out_channel)
        self.en4 = ChannelAttention(out_channel)
        self.conv5 = self.convlayer(out_channel, out_channel, 3)
        self.e5 = ChannelAttention(out_channel)
        self.HSI5 = HSI_block(out_channel, out_channel)
        self.en5 = ChannelAttention(out_channel)
        self.conv6 = self.convlayer(out_channel, out_channel, 3)
        self.e6 = ChannelAttention(out_channel)
        self.HSI6 = HSI_block(out_channel, out_channel)
        self.en6 = ChannelAttention(out_channel)
        self.conv7 = self.convlayer(out_channel, out_channel, 3)
        self.e7 = ChannelAttention(out_channel)
        self.HSI7 = HSI_block(out_channel, out_channel)
        self.en7 = ChannelAttention(out_channel)
        self.conv8 = self.convlayer(out_channel, out_channel, 3)
        self.e8 = ChannelAttention(out_channel)
        self.HSI8 = HSI_block(out_channel, out_channel)
        self.en8 = ChannelAttention(out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        gelu = nn.LeakyReLU()
        layers = filter(lambda x: x is not None, [pader, conver, gelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        B = self.conv_B(x)
        G = self.conv_G(x)
        R = self.conv_R(x)
        B = self.convinputb(B)
        G = self.convinputg(G)
        R = self.convinputr(R)
        f0 = torch.cat([B, G], 1)
        f0 = torch.cat([f0, R], 1)
        Sf0 = self.HSI1(f0)
        e1 = self.e1(f0)
        en1 = self.en1(f0)
        f0_e = f0 * e1
        Sf0_en = Sf0 * en1
        f0_conv = self.conv1(f0)
        f1 = torch.add(f0_conv, f0_e)
        f1 = torch.add(f1, Sf0_en)
        Sf1 = self.HSI2(f1)
        e2 = self.e2(f1)
        en2 = self.en2(f1)
        f1_e = f0 * e2
        Sf1_en = Sf1 * en2
        f1_conv = self.conv2(f1)
        f2 = torch.add(f1_conv, f1_e)
        f2 = torch.add(f2, Sf1_en)
        Sf2 = self.HSI3(f2)
        e3 = self.e3(f2)
        en3 = self.en3(f2)
        f2_e = f0 * e3
        Sf2_en = Sf2 * en3
        f2_conv = self.conv3(f2)
        f3 = torch.add(f2_conv, f2_e)
        f3 = torch.add(f3, Sf2_en)
        Sf3 = self.HSI4(f3)
        e4 = self.e4(f3)
        en4 = self.en4(f3)
        f3_e = f0 * e4
        Sf3_en = Sf3 * en4
        f3_conv = self.conv4(f3)
        f4 = torch.add(f3_conv, f3_e)
        f4 = torch.add(f4, Sf3_en)
        Sf4 = self.HSI5(f4)
        e5 = self.e5(f4)
        en5 = self.en5(f4)
        f4_e = f0 * e5
        Sf4_en = Sf4 * en5
        f4_conv = self.conv5(f4)
        f5 = torch.add(f4_conv, f4_e)
        f5 = torch.add(f5, Sf4_en)
        Sf5 = self.HSI6(f5)
        e6 = self.e6(f5)
        en6 = self.en6(f5)
        f5_e = f0 * e6
        Sf5_en = Sf5 * en6
        f5_conv = self.conv6(f5)
        f6 = torch.add(f5_conv, f5_e)
        f6 = torch.add(f6, Sf5_en)
        Sf6 = self.HSI7(f6)
        e7 = self.e7(f6)
        en7 = self.en7(f6)
        f6_e = f0 * e7
        Sf6_en = Sf6 * en7
        f6_conv = self.conv7(f6)
        f7 = torch.add(f6_conv, f6_e)
        f7 = torch.add(f7, Sf6_en)
        Sf7 = self.HSI8(f7)
        e8 = self.e8(f7)
        en8 = self.en8(f7)
        f7_e = f0 * e8
        Sf7_en = Sf7 * en8
        f7_conv = self.conv8(f7)
        f8 = torch.add(f7_conv, f7_e)
        f8 = torch.add(f8, Sf7_en)

        return f8

class HSI_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(HSI_block, self).__init__()
        self.conv1 = self.convlayer(in_channels,128,3)
        self.conv2 = nn.Conv2d(128, out_channels, 3, stride=1,padding=1, bias=True)
        self.conv3 = self.convlayer(out_channels,out_channels,1)
        self.gelu = nn.LeakyReLU()

    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        gelu = nn.LeakyReLU(0.2, inplace=True)
        layers = filter(lambda x: x is not None, [pader, conver, gelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        residuals=torch.add(x,out2)
        residuals=self.gelu(residuals)
        output=self.conv3(residuals)
        return output