import torch
import torch.nn as nn
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
        model = DenseU(31,31)
    elif method == 'CanNet':
        model = Cannet(31,31)
    elif method == 'HSCNN+':
        model = hscnn_plus(38,in_channels=31, out_channels=31)
    elif method == 'sRCNN':
        model = sRCNN(31,31,16,128,3,1)
    elif method == 'AWAN':
        model = AWAN(inplanes=31, planes=31,reduction=3)
    elif method == 'FMNet':
        model = FMNet(bNum=3, nblocks=4, input_features=31, num_features=31, out_features=31)
    elif method == 'HRNet':
        model = SGN(in_channels=31, out_channels=31)
    elif method =='HSRnet':
        model=HSRnet(31,31)
    elif method == 'HSACS':
        model =HSACS(inplanes=31, planes=31)
    elif method =='GDNet':
        model=GDNet(31,31)
    elif method =='SSDCN':
        model=SSDCN(31,31)
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model