import torch
import torch.nn as nn
from math import sqrt

class Exblock(nn.Module):
    def __init__(self, in_channels):
        super(Exblock, self).__init__()
        self.conv1 = self.convlayer(in_channels, 64, 1,1)
        self.conv3_1 = self.convlayer(64, 16, 3,1)
        self.conv3_2 = self.convlayer(64, 16, 3,1)
        self.conv1_1 = self.convlayer(16, 8, 1,1)
        self.conv1_2 = self.convlayer(16, 8, 1,1)
        self.conv1_final = self.convlayer(48, 16, 1,1)

    def convlayer(self, in_channels, out_channels, kernel_size,stride):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)
        relu = nn.ReLU()
        layers = filter(lambda x: x is not None, [pader, conver, relu])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        f3_1=self.conv3_1(out1)
        f3_2 = self.conv3_2(out1)
        f1_1 = self.conv1_1(f3_1)
        f1_2 = self.conv1_2(f3_2)
        f=torch.cat([f3_1,f3_2,f1_1,f1_2],1)
        out=self.conv1_final(f)
        output=torch.cat([x,out],1)
        return output

class hscnn_plus(nn.Module):
    def __init__(self, n,in_channels,out_channels):
        super(hscnn_plus, self).__init__()
        self.conv3_1 = self.convlayer(in_channels, 16, 3, 1)
        self.conv3_2 = self.convlayer(in_channels, 16, 3, 1)
        self.conv1_1 = self.convlayer(16, 16, 1, 1)
        self.conv1_2 = self.convlayer(16, 16, 1, 1)
        self.conv = self.Exlayer(n,64)
        self.conv1_final = self.convlayer(64+n*16, out_channels, 1, 1)

    def convlayer(self, in_channels, out_channels, kernel_size, stride):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)
        relu = nn.ReLU()
        layers = filter(lambda x: x is not None, [pader, conver, relu])
        return nn.Sequential(*layers)

    def Exlayer(self, n,in_channels):
        main = nn.Sequential()
        for i in range(n):
            name='Ex'+str(i)
            conv=Exblock(in_channels+16*i)
            main.add_module(name,conv)
        return main

    def forward(self, x):
        f3_1=self.conv3_1(x)
        f3_2 =self.conv3_2(x)
        f1_1=self.conv1_1(f3_1)
        f1_2 = self.conv1_1(f3_2)
        f_in=torch.cat([f3_1,f3_2,f1_1,f1_2],1)
        f_out = self.conv(f_in)
        output=self.conv1_final(f_out)
        return output
