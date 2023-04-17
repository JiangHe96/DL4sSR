import torch
import torch.nn as nn

class Resb(nn.Module):
    def __init__(self, in_f,out_f,k,p):
        super(Resb, self).__init__()
        self.conv1 = nn.Conv3d(in_f,out_f,kernel_size=(k,1,1),padding=(p,0,0))
        self.PRelu = nn.PReLU()

    def forward(self, x):
        residuals = self.PRelu(self.conv1(x))
        out=residuals+x
        return out

class sRCNN(nn.Module):
    def __init__(self, in_channel,out_channel,L,F,K,P):
        super(sRCNN, self).__init__()
        self.conv_t = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.conv1 = nn.Conv3d(1, F, kernel_size=(K, 1, 1), padding=(P, 0, 0))
        self.resblock = self.Reslayer(L,F,F,K,P)
        self.conv_final = nn.Conv3d(F, 1, 1,1)
        self.PRelu = nn.PReLU()

    def Reslayer(self, n,in_f,out_f,k,p):
        main = nn.Sequential()
        for i in range(n):
            name='Resb'+str(i)
            conv=Resb(in_f,out_f,k,p)
            main.add_module(name,conv)
        return main

    def forward(self, x):
        HSI_original=self.PRelu(self.conv_t(x))
        f1=self.PRelu(self.conv1(HSI_original.unsqueeze(1)))
        residuls=self.resblock(f1)
        f2=residuls+f1
        output = self.PRelu(self.conv_final(f2))
        return output.squeeze(1)
