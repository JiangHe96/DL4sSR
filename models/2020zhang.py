import torch
import torch.nn as nn
from math import sqrt


class Conv_ReLU_Block(nn.Module):
    def __init__(self, nFeat=64, ksize=3):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=ksize, stride=1,
                              padding=int((ksize - 1) / 2), bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Residual_Block(nn.Module):
    def __init__(self, Cn=64, ksize=3):
        super(Residual_Block, self).__init__()
        self.conv = self.make_layer(Conv_ReLU_Block, conv_num=1, cn=Cn)
        self.ouput = nn.Conv2d(in_channels=Cn, out_channels=Cn, kernel_size=ksize, stride=1,
                               padding=int((ksize - 1) / 2), bias=False)
        self.relu = nn.ReLU(inplace=True)

    def make_layer(self, block, conv_num, cn):
        layer = []
        for _ in range(conv_num):
            layer.append(block(cn))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv(x)
        out = self.ouput(out)
        out = out + x
        # out = self.relu(out)
        return out


class Representation(nn.Module):
    def __init__(self, inFeat=3, nFeat=64, layers=3, outChn=5):
        super(Representation, self).__init__()
        self.input = nn.Conv2d(in_channels=inFeat, out_channels=nFeat, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = self.make_layer(Residual_Block, conv_num=layers, cn=nFeat)
        self.conv2 = self.make_layer(Conv_ReLU_Block, conv_num=layers, cn=nFeat)
        self.output = nn.Conv2d(in_channels=nFeat, out_channels=outChn, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.softmax = nn.Softmax2d()
        self.map = nn.Sigmoid()

        # filter initializiation ?
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, conv_num, cn):
        layer = []
        for _ in range(conv_num):
            layer.append(block(cn))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.relu(self.input(x))
        out = self.conv2(out)
        out = self.output(out)
        out = self.map(out)
        out_s = torch.sum(out, dim=1, keepdim=True)
        out = out / (out_s + 1e-9)
        out = out.view(out.shape[0], out.shape[1], 1, out.shape[2], out.shape[3])
        return out


class Atom(nn.Module):
    def __init__(self, inChn=64, nFeat=64, outChn=3, layers=3, ksize=3):
        super(Atom, self).__init__()
        self.input = nn.Conv2d(inChn, nFeat, kernel_size=ksize, padding=int((ksize - 1) / 2), bias=True)
        self.map = self.make_layer(Conv_ReLU_Block, conv_num=layers - 2, cn=nFeat, ksize=ksize)
        self.output = nn.Conv2d(nFeat, outChn, kernel_size=ksize, padding=int((ksize - 1) / 2), bias=True)
        self.relu = nn.ReLU(inplace=True)

    def make_layer(self, block, conv_num, cn, ksize):
        layer = []
        for _ in range(conv_num):
            layer.append(block(cn, ksize))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.relu(self.input(x))
        out = self.map(out)
        out = self.output(out)
        return out


class Dictionary(nn.Module):
    def __init__(self, inChn=64, nFeat=64, outChn=3, layers=3, bNum=5):
        super(Dictionary, self).__init__()
        self.base = nn.ModuleList([Atom(inChn=inChn, nFeat=nFeat, outChn=outChn, layers=layers, ksize=3)])
        for i in range(bNum - 1):
            self.base.append(Atom(inChn=inChn, nFeat=nFeat, outChn=outChn, layers=layers, ksize=3 + (i + 1) * 4))

    def forward(self, x):
        for i, conv in enumerate(self.base):
            temp = conv(x)
            temp = temp.view(temp.shape[0], 1, temp.shape[1], temp.shape[2], temp.shape[3])
            if i > 0:
                out = torch.cat([out, temp], dim=1)
            else:
                out = temp

        return out


class FMNet(nn.Module):
    def __init__(self, init=True, bNum=3, nblocks=4, input_features=31, num_features=64, out_features=3):
        super(FMNet, self).__init__()
        self.input = nn.Conv2d(in_channels=input_features, out_channels=num_features, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bNum = bNum
        self.nblocks = nblocks

        self.convs = nn.ModuleList(
            [self.make_layer(Conv_ReLU_Block, conv_num=1, cn=num_features) for i in range(self.nblocks)])
        self.dics = nn.ModuleList(
            [Dictionary(inChn=num_features, nFeat=num_features, outChn=num_features, layers=2, bNum=bNum) for i in
             range(self.nblocks)])
        self.reps = nn.ModuleList(
            [Representation(inFeat=num_features, nFeat=num_features, layers=3, outChn=bNum) for i in
             range(self.nblocks)])

        self.cdic = Dictionary(inChn=num_features * self.nblocks, nFeat=num_features, outChn=num_features, layers=2,
                               bNum=bNum)
        self.crep = Representation(inFeat=num_features * self.nblocks, nFeat=num_features, layers=3, outChn=bNum)

        self.odic = Dictionary(inChn=num_features, nFeat=num_features, outChn=out_features, layers=2, bNum=bNum)
        self.orep = Representation(inFeat=num_features, nFeat=num_features, layers=3, outChn=bNum)

        self.relu = nn.ReLU(inplace=True)

        if init == True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, conv_num=2, cn=64, ksize=3):
        layer = []
        for _ in range(conv_num):
            layer.append(block(cn, ksize))
        return nn.Sequential(*layer)

    def forward(self, x):

        fea_0 = self.input(x)
        fea = fea_0

        for i in range(self.nblocks):

            fea = self.convs[i](fea)
            wei_ = self.reps[i](fea)
            fea_a = self.dics[i](fea)

            fea_a = fea_a * wei_
            fea = torch.sum(fea_a, dim=1)

            if i > 0:
                weis_out = torch.cat([weis_out, wei_], dim=1)
                fea_out = torch.cat([fea_out, fea], dim=1)
            else:
                weis_out = wei_
                fea_out = fea

        wei_ = self.crep(fea_out)
        fea_a = self.cdic(fea_out)

        fea_a = fea_a * wei_
        fea = torch.sum(fea_a, dim=1)

        weis_out = torch.cat([weis_out, wei_], dim=1)

        wei_ = self.orep(fea)
        fea_a = self.odic(fea)

        fea_a = fea_a * wei_
        fea = torch.sum(fea_a, dim=1)
        out = fea + x

        weis_out = torch.cat([weis_out, wei_], dim=1)
        return out