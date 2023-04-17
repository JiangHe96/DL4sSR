import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class My_Bn_1(nn.Module):
    def __init__(self):
        super(My_Bn_1, self).__init__()

    def forward(self, x):
        # print(x.shape)
        # _,C,_,_ = x.shape
        # x1,x2 = torch.split(x,C//2,dim=1)
        # x1 = x1 - nn.AdaptiveAvgPool2d(1)(x1)
        # x1 = torch.cat([x1,x2],dim=1)
        # x =
        return x - torch.mean(x, dim=1, keepdim=True)


class My_Bn_2(nn.Module):
    def __init__(self):
        super(My_Bn_2, self).__init__()

    def forward(self, x):
        # print(x.shape)
        # _,C,_,_ = x.shape
        # x1,x2 = torch.split(x,C//2,dim=1)
        # x1 = x1 - nn.AdaptiveAvgPool2d(1)(x1)
        # x1 = torch.cat([x1,x2],dim=1)
        # x =
        return x - nn.AdaptiveAvgPool2d(1)(x)


class BCR(nn.Module):
    def __init__(self, kernel, cin, cout, group=1, stride=1, RELU=True, padding=0, BN=False, spatial_norm=False,
                 bias=False):
        super(BCR, self).__init__()
        if stride > 0:
            self.conv = nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=kernel, groups=group, stride=stride,
                                  padding=padding, bias=bias)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=cin, out_channels=cout, kernel_size=kernel, groups=group,
                                           stride=int(abs(stride)), padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.Swish = MemoryEfficientSwish()

        if RELU:
            if BN:
                if spatial_norm:
                    self.Bn = My_Bn_2()
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                        self.relu,
                    )
                else:
                    self.Bn = My_Bn_1()
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                        self.relu,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                    self.relu
                )
        else:
            if BN:
                if spatial_norm:
                    self.Bn = My_Bn_2()
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                    )
                else:
                    self.Bn = My_Bn_1()
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                        self.relu,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                )

    def forward(self, x):
        output = self.Module(x)
        return output


class denselayer(nn.Module):
    def __init__(self, cin, cout=31, RELU=True, BN=True, bias=True):
        super(denselayer, self).__init__()
        self.compressLayer = BCR(kernel=1, cin=cin, cout=cout, RELU=RELU, BN=BN, spatial_norm=False, bias=bias)
        self.actlayer = BCR(kernel=3, cin=cout, cout=cout, group=cout, RELU=RELU, padding=1, BN=BN, spatial_norm=False,
                            bias=bias)

    def forward(self, x):
        output = self.compressLayer(x)
        output = self.actlayer(output)

        return output


class stage(nn.Module):
    def __init__(self, cin, cout, final=False, BN=False, linear=True, bias=False):
        super(stage, self).__init__()
        self.Upconv = BCR(kernel=3, cin=cin, cout=cout, stride=1, padding=1, RELU=False, BN=False)
        if final == True:
            f_cout = cout + 1
        else:
            f_cout = cout
        mid = cout * 3
        self.denselayers = nn.ModuleList([
            denselayer(cin=1 * cout, cout=cout * 2, BN=BN, bias=bias),
            denselayer(cin=3 * cout, cout=cout * 2, BN=BN, bias=bias),
            denselayer(cin=5 * cout, cout=cout * 2, BN=BN, bias=bias),
            denselayer(cin=7 * cout, cout=cout * 2, BN=BN, bias=bias),
            denselayer(cin=9 * cout, cout=f_cout, RELU=False, BN=False, bias=bias)])
        self.bn = My_Bn_1()
        self.linear = linear

    def forward(self, MSI):
        MSI = self.Upconv(MSI)
        # print('-----------ERROR------------')
        # print(MSI.shape)
        x = [MSI]
        # print(MSI.shape)
        for layer in self.denselayers:
            x_ = layer(torch.cat(x, 1))
            x.append(x_)
        if self.linear == True:
            return x[-1] + MSI
        else:
            return x[-1]


class reconnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(reconnet, self).__init__()

        self.stages = nn.ModuleList([
            stage(cin=in_ch, cout=out_ch, BN=False, linear=False, bias=True),
            stage(cin=in_ch, cout=out_ch, BN=True, linear=True, bias=False),
            stage(cin=in_ch, cout=out_ch, BN=True, linear=True, bias=False),
            stage(cin=in_ch, cout=out_ch, BN=True, linear=True, bias=False),
            stage(cin=in_ch, cout=out_ch, BN=True, linear=True, bias=False),
            stage(cin=in_ch, cout=out_ch, BN=True, linear=True, bias=False),
            stage(cin=in_ch, cout=out_ch, BN=True, linear=True, bias=False),
            stage(cin=in_ch, cout=out_ch, BN=True, linear=True, bias=False),
            # stage(cin=3,cout=31,extra = extra[1],BN = True,linear=True,bias=False),
            # stage(cin=3,cout=31,extra = extra[1],BN = True,linear=True,bias=False),
            # stage(cin=3,cout=31,extra = extra[1],BN = True,linear=True,bias=False),
            # stage(cin=3,cout=31,extra = extra[1],BN = True,linear=True,bias=False),
            # stage(cin=3,cout=31,extra = extra[1],BN = True,linear=True,bias=False),
            # stage(cin=3,cout=31,extra = extra[1],BN = True,linear=True,bias=False),
            stage(cin=in_ch, cout=out_ch,  BN=True, linear=True, bias=False)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.degradation = BCR(kernel=3, cin=out_ch, cout=in_ch, group=1, stride=1, RELU=False, BN=False, padding=1, bias=False)

    def forward(self, MSI, extra_data=None):

        ref = [np.array(range(8)) * 4, np.array(range(16)) * 2]
        ref[0][-1] = 30
        ref[1][-1] = 30
        recon_out = None
        MSI = [MSI]
        for index, stage in enumerate(self.stages):
            recon = stage(MSI[-1])
            if recon_out is None:
                recon_out = recon
            else:
                recon_out = recon_out + recon
            msi_ = MSI[0] - self.degradation(recon_out)
            MSI.append(msi_)

        return recon_out