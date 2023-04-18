import torch
import torch.nn as nn
import numpy as np

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()

    def forward(self, im_fake, im_true):
        sum1 = torch.sum(im_true * im_fake, 1)
        sum2 = torch.sum(im_true * im_true, 1)
        sum3 = torch.sum(im_fake * im_fake, 1)
        t = (sum2 * sum3) ** 0.5
        numlocal = torch.gt(t, 0)
        num = torch.sum(numlocal)
        t = sum1 / t
        angle = torch.acos(t)
        sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle)
        SAM = sumangle * 180 / 3.14159256
        return SAM

from store2tiff import readTiff
from torch.autograd import Variable
import matplotlib.pyplot as plt
import h5py
import cv2
criterion_sam = Loss_SAM()
outputpath=r'.\DL4sSR Review\latex\IF\sr\sam'+'/'
selectedimg=[901,903,908,912,919,930,938,944,946,948]
for imgnum in selectedimg:
    data= h5py.File(r'.\dataset\Train_Spec'+'/ARAD_1K_0'+str(imgnum)+'.mat','r')
    GT=data['cube']
    GT = Variable(torch.from_numpy(GT[()]).float()).view(1, -1, GT.shape[1], GT.shape[2])
    GT = GT.permute(0, 1, 3, 2)
    name=['AWAN','CanNet','DU','FMNet','GDNet','HRNet','HSACS','HSCNN+','HSRnet','sRCNN','SSDCN']
    filename=['AWAN_b8_adalr0.0001','CanNet_b8_adalr0.0005','DenseUnet_b8_adalr0.0002','FMNet_b8_adalr0.0001','GDNet_b8_adalr0.001','HRNet_b8_adalr0.0001','HSACS_b8_adalr5e-05','HSCNN+_b8_adalr0.0002','HSRnet_b8_adalr0.0001','sRCNN_b8_adalr0.0001','SSDCN_b8_adalr0.01']

    for i,fi in zip(name, filename):
        # output, _, _ = readTiff('%s%s%s' % ('E:/6DL4sSR/2SR\show/tiff2/',i ,'.tif'))
        # output, _, _ = readTiff('E:/6DL4sSR/2SR\show/tiff2/ARAD_1K_945.tif')
        data = h5py.File(r'E:\6DL4sSR\2SR\output/' + fi+'/ARAD_1K_' + str(imgnum) + '.mat','r')
        output = data['cube']
        output=Variable(torch.from_numpy(output[()]).float()).view(1, -1, output.shape[1], output.shape[2])
        output=output.permute(0,1,3,2)
        sam=criterion_sam(output[:,:,:480,:],GT[:,:,:480,:])
        vmin=0
        vmax=15
        cm='jet'

        plt.imsave('%s%s%d%s%s%s' % (outputpath,'/',imgnum,'/', i,'.png'),sam[0,:,:],cmap = (cm), vmin=vmin, vmax=vmax)
