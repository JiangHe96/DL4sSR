import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f
import torch
import torch.nn as nn
from math import sqrt

class SSDCN(nn.Module):
    def __init__(self,in_channels,out_channels,NNN = 128):
        super(SSDCN, self).__init__()
        # to extract intial feature
        self.Spe1_conv = nn.Conv2d(in_channels, NNN, 1)
        # ------------------residual spectral-spatial blocks 1------------------
        # 1. 1x1 spectral branch
        self.Spe2_conv = nn.Conv2d(NNN, NNN, 1)
        self.Spe3_conv = nn.Conv2d(NNN, NNN, 1)
        self.Spe3_sea = SEA(NNN)
        # ------------------residual spectral-spatial blocks 2------------------
        # 2. 1x1 spectral branch
        self.Spe4_conv = nn.Conv2d(NNN, NNN, 1)
        self.Spe5_conv = nn.Conv2d(NNN, NNN, 1)
        self.Spe5_sea = SEA(NNN)
        # ------------------residual spectral-spatial blocks 3------------------
        # 3. 3x3 spatial branch
        self.Spe6_conv = nn.Conv2d(NNN, NNN, 3)
        self.Spe7_conv = nn.Conv2d(NNN, NNN, 3)
        self.padding = nn.ReflectionPad2d(1)
        self.Spe7_sea = SEA(NNN)
        # ------------------residual spectral-spatial blocks 4------------------
        # 4. 3x3 spatial branch
        self.Spe8_conv = nn.Conv2d(NNN, NNN, 1)
        self.Spe9_conv = nn.Conv2d(NNN, NNN, 1)
        self.Spe9_sea = SEA(NNN)
        # Out
        self.out = nn.Conv2d(NNN, out_channels, 1)

        # ----------------------------------------Dual Network--------------------------------------------
        self.down = nn.Conv2d(out_channels, in_channels, 1)
        self.relu=nn.ReLU()

    def forward(self, x):
        Spe1=self.Spe1_conv(x)
        # ------------------residual spectral-spatial blocks 1------------------
        # 1. 1x1 spectral branch
        Spe2 = self.Spe2_conv(Spe1)
        Spe3 = self.Spe3_conv(Spe2)
        Spe3 = self.Spe3_sea(Spe3)
        Spe3_residual =self.relu(Spe3+Spe1)

        # ------------------residual spectral-spatial blocks 2------------------
        # 2. 1x1 spectral branch
        Spe4 = self.Spe4_conv(Spe3_residual)
        Spe5 = self.Spe5_conv(Spe4)
        Spe5 = self.Spe5_sea(Spe5)
        Spe5_residual = self.relu(Spe5+Spe3_residual)

        # ------------------residual spectral-spatial blocks 3------------------
        # 3. 3x3 spatial branch
        Spe6 = self.Spe6_conv(self.padding(Spe5_residual))
        Spe7 = self.Spe7_conv(self.padding(Spe6))
        Spe7 = self.Spe7_sea(Spe7)
        Spe7_residual = self.relu(Spe7 + Spe5_residual)

        # ------------------residual spectral-spatial blocks 4------------------
        # 4. 3x3 spatial branch
        Spe8 = self.Spe8_conv(Spe7_residual)
        Spe9 = self.Spe9_conv(Spe8)
        Spe9 = self.Spe9_sea(Spe9)
        Spa9_residual = self.relu(Spe9 + Spe7_residual)

        # Out
        Output_HSI = self.out(Spa9_residual)

        # ----------------------------------------Dual Network--------------------------------------------
        Output_MSI = self.down(Output_HSI)
        return Output_HSI

class SEA(nn.Module):
    def __init__(self, in_planes):
        super(SEA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, 16, 1, bias=False)
        self.relu1 =nn.ReLU()
        self.fc2 = nn.Conv2d(16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = self.sigmoid(avg_out)*x
        return out