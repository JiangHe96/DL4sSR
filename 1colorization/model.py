import torch
import importlib

import sys
sys.path.append("../models")

net=importlib.import_module('models.2017galliani')
DenseU=net.DenseU
net=importlib.import_module('models.2018can')
Cannet=net.Cannet
net=importlib.import_module('models.2018shi')
hscnn_plus=net.hscnn_plus
net=importlib.import_module('models.2019gewali')
sRCNN=net.sRCNN
net=importlib.import_module('models.2020li2')
AWAN=net.AWAN
net=importlib.import_module('models.2020zhang')
FMNet=net.FMNet
net=importlib.import_module('models.2020zhao')
SGN=net.SGN
net=importlib.import_module('models.2021he')
HSRnet=net.HSRnet
net=importlib.import_module('models.2021he_color')
HSRnet_color=net.HSRnet
net=importlib.import_module('models.2021li')
HSACS=net.HSACS
net=importlib.import_module('models.2021zhu')
GDNet=net.reconnet
net=importlib.import_module('models.2022chen')
SSDCN=net.SSDCN

def model_generator(method, pretrained_model_path=None):
    if method == 'DenseUnet':
        model = DenseU(1,3)
    elif method == 'CanNet':
        model = Cannet(1,3)
    elif method == 'HSCNN+':
        model = hscnn_plus(38,in_channels=1, out_channels=3)
    elif method == 'sRCNN':
        model = sRCNN(1,3,16,128,3,1)
    elif method == 'AWAN':
        model = AWAN(inplanes=1, planes=3,reduction=3)
    elif method == 'FMNet':
        model = FMNet(bNum=3, nblocks=4, input_features=1, num_features=31, out_features=3)
    elif method == 'HRNet':
        model = SGN(in_channels=1, out_channels=3)
    elif method =='HSRnet_color':
        model=HSRnet_color(1,3)
    elif method == 'HSACS':
        model =HSACS(inplanes=1, planes=3)
    elif method =='GDNet':
        model=GDNet(1,3)
    elif method =='SSDCN':
        model=SSDCN(1,3)
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
