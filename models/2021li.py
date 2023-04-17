import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)


def conv3x3x3(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)


class STA2D(nn.Module):
    def __init__(self, channel=256, reduction=16):
        super(STA2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # kernel_size = 1
        # self.spatial = nn.Conv2d(channel, channel, kernel_size, stride=1, padding=(kernel_size-1) // 2)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.count_grad(x)
        b, c, _, _ = out.size()
        y = self.avg_pool(out).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        z = self.count_struct_tensor(x)

        out = x * z + x * y + x
        return out

    def count_struct_tensor(self, outputs):
        b, c, h, w, = outputs.shape
        outputs = outputs.view(b * c, h, w).unsqueeze(0).unsqueeze(0)
        gx_kernel = torch.Tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).cuda()
        gy_kernel = torch.Tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).cuda()
        gradx = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False).cuda()
        gradx.weight.data = gx_kernel.view(1, 1, 1, 3, 3)
        grady = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False).cuda()
        grady.weight.data = gy_kernel.view(1, 1, 1, 3, 3)
        with torch.no_grad():
            imx = gradx(outputs)
            imy = grady(outputs)
        M00, M01, M11 = imx * imx, imx * imy, imy * imy
        outputs_e1 = (M00 + M11) / 2 + torch.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
        # outputs_e1 = torch.exp(outputs_e1)
        outputs_e1 = outputs_e1 / torch.max(outputs_e1)
        return outputs_e1.view(b, c, h, w)

    def count_grad(self, outputs):
        # n - 1
        grad_outputs_n_1 = outputs[:, 1:, :, :] - outputs[:, :-1, :, :]
        # 1 - n
        grad_outputs_1_n = outputs[:, :-1, :, :] - outputs[:, 1:, :, :]
        grad_outputs = torch.cat((2*grad_outputs_1_n[:, 0:1, :, :],
                                  grad_outputs_1_n[:, 1:, :, :]+grad_outputs_n_1[:, :-1, :, :],
                                  2*grad_outputs_n_1[:, -1:, :, :]), 1)
        return grad_outputs


class STA3D(nn.Module):
    def __init__(self, channel=24, reduction=8):
        super(STA3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # kernel_size = 1
        # self.spatial = nn.Sequential(nn.Conv3d(channel, channel, kernel_size, stride=1, padding=(kernel_size-1) // 2))
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.count_grad(x)
        b, c, d, h, w = out.size()
        y = self.avg_pool(out.view(b, c*d, h, w)).view(b, c*d)
        y = self.fc(y).view(b, c, d, 1, 1)
        z = self.count_struct_tensor(x)
        out = x * z + x * y + x
        return out

    def count_struct_tensor(self, outputs):
        b, c, d, h, w, = outputs.shape
        outputs = outputs.view(b * c * d, h, w).unsqueeze(0).unsqueeze(0)
        gx_kernel = torch.Tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).cuda()
        gy_kernel = torch.Tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).cuda()
        gradx = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                          bias=False).cuda()
        gradx.weight.data = gx_kernel.view(1, 1, 1, 3, 3)
        grady = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                          bias=False).cuda()
        grady.weight.data = gy_kernel.view(1, 1, 1, 3, 3)
        with torch.no_grad():
            imx = gradx(outputs)
            imy = grady(outputs)
        M00, M01, M11 = imx * imx, imx * imy, imy * imy
        outputs_e1 = (M00 + M11) / 2 + torch.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
        # outputs_e1 = torch.exp(outputs_e1)
        outputs_e1 = outputs_e1 / torch.max(outputs_e1)
        return outputs_e1.view(b, c, d, h, w)

    def count_grad(self, outputs):
        # n - 1
        grad_outputs_n_1 = outputs[:, :, 1:, :, :] - outputs[:, :, :-1, :, :]
        # 1 - n
        grad_outputs_1_n = outputs[:, :, :-1, :, :] - outputs[:, :, 1:, :, :]

        grad_outputs = torch.cat((2*grad_outputs_1_n[:, :, 0:1, :, :],
                                  grad_outputs_1_n[:, :, 1:, :, :]+grad_outputs_n_1[:, :, :-1, :, :],
                                  2*grad_outputs_n_1[:, :, -1:, :, :]), 2)

        return grad_outputs


class ResBlock2D(nn.Module):
    def __init__(self, channel,in_channels=256, out_channels=256):
        super(ResBlock2D, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.relu1 = nn.PReLU()
        self.conv2 = conv3x3(in_channels, out_channels)
        self.relu2 = nn.PReLU()
        self.sta = STA2D()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.sta(out)
        out = torch.add(out, residual)
        out = self.relu2(out)
        return out


class ResBlock3D(nn.Module):
    def __init__(self, channel,in_channels=8, out_channels=8):
        super(ResBlock3D, self).__init__()
        self.conv1 = conv3x3x3(in_channels, out_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = conv3x3x3(in_channels, out_channels)
        self.prelu2 = nn.PReLU()
        self.sta = STA3D(channel*8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.sta(out)
        out = torch.add(out, x)
        out = self.prelu2(out)
        return out


class HSACS(nn.Module):
    def __init__(self, inplanes=3, planes=31, block2D=ResBlock2D, block3D=ResBlock3D, layers=[16, 4]):
        super(HSACS, self).__init__()
        # 2D Nets
        self.planes=planes
        self.conv2D_head = conv3x3(inplanes, 256)
        self.ResNet2D = self.make_layer(block2D, layers[0])
        self.conv2D_tail = conv3x3(256, 256)
        self.prelu2D = nn.PReLU()
        self.output_conv2D = conv3x3(256, planes)
        # 3D Nets
        self.conv3D_head = conv3x3x3(1, 8)
        self.ResNet3D = self.make_layer(block3D, layers[1])
        self.conv3D_tail = conv3x3x3(8, 8)
        self.prelu3D = nn.PReLU()
        self.output_conv3D = conv3x3x3(8, 1)
        # 2D-3D Residual
        self.output_conv = conv3x3(planes, planes)

    def make_layer(self, block, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(block(channel=self.planes))  # there is a ()
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.DRN2D(x)
        residual = out
        out = self.DRN3D(out)
        out = self.output_conv(residual + out)
        return out

    def DRN2D(self, x):
        out = self.conv2D_head(x)
        residual = out
        out = self.ResNet2D(out)
        out = self.conv2D_tail(out)
        out = torch.add(out, residual)
        out = self.prelu2D(out)
        out = self.output_conv2D(out)
        return out

    def DRN3D(self, x):
        out = self.conv3D_head(x.unsqueeze(1))
        residual = out
        out = self.ResNet3D(out)
        out = self.conv3D_tail(out)
        out = torch.add(out, residual)
        out = self.prelu3D(out)
        out = self.output_conv3D(out)
        return out.squeeze(1)