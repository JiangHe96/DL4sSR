import torch
import torch.nn as nn
from math import sqrt

class Cannet(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Cannet, self).__init__()
        self.conv1 = self.convlayer(in_channel, 128, 5)
        self.conv_residual = self.convlayer(in_channel, out_channel, 7)
        self.conv2 = self.convlayer(128, 32, 1)
        self.resB1=Resblock(32)
        self.resB2=Resblock(32)
        self.conv5 = self.convlayer(64, 128, 1)
        self.conv6 = self.convlayer(128, out_channel, 5)


    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        layers = filter(lambda x: x is not None, [pader, conver])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1=self.conv1(x)
        out2=self.conv2(out1)
        out3=self.resB1(out2)
        out4=self.resB2(out3)
        out4=torch.cat([out4,out2],1)
        out5=self.conv5(out4)
        out6=self.conv6(out5)
        residual=self.conv_residual(x)
        output=torch.add(out6,residual)
        return output

class Resblock(nn.Module):
    def __init__(self, in_channels):
        super(Resblock, self).__init__()
        self.conv1 = self.convlayer(in_channels, 32, 3)
        self.conv2 = self.convlayer(in_channels, 32, 3)
        self.prelu=nn.PReLU()

    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        layers = filter(lambda x: x is not None, [pader, conver])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        residual=self.conv2(out1)
        out2=torch.add(x,residual)
        output=self.prelu(out2)
        return output
