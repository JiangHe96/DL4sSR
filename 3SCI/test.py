import argparse, os
import torch
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage as io
import scipy.io as sio
import sys
sys.path.append("..")
import evaluate
from store2tiff import writeTiff as wtiff


def generate_masks(channel,batch_size):
    mask = sio.loadmat('mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:,:,np.newaxis],(1,1,channel))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).float()
    return mask3d_batch

def gen_meas_torch(data_batch, mask3d_batch, is_training=False):
    nC = data_batch.shape[1]
    if is_training is False:
        [batch_size, nC, H, W] = data_batch.shape
        mask3d_batch = (mask3d_batch[0,:,:,:]).expand([batch_size, nC, H, W]).cuda().float()
    temp = shift(mask3d_batch*data_batch, 2)
    meas = torch.sum(temp, 1)/nC*2          # meas scale
    y_temp = shift_back(meas,nC)
    PhiTy = torch.mul(y_temp, mask3d_batch)
    return PhiTy

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col+(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,step*i:step*i+col] = inputs[:,i,:,:]
    return output

def shift_back(inputs,nC,step=2):          # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col-(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,:] = inputs[:,:,step*i:step*i+col-(nC-1)*step]
    return output

mask3d_batch = generate_masks(31, 1)

parser = argparse.ArgumentParser(description="DL4sSR Spectral Compressive Imaging Demo")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda?")
parser.add_argument("--scale", default=1, type=int, help="scale factor")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--name", default="FMNet_step10_b1_adam_L1loss_adalr0.0001", type=str, help="model name")
opt = parser.parse_args()
cuda = opt.cuda
name=opt.name
outfile='output/'+opt.name+'/'
if not os.path.exists(outfile):
    os.makedirs(outfile)
data = io.loadmat('CAVE_6test.mat')
rgb = data['data']

rad=data['label']
ratio=2
band_num=rad.shape[2]
ite=np.ones(rad.shape[3],int)*200
index=torch.zeros((5,band_num+1,len(ite)))
for i in range(rad.shape[3]):
    test_ite = ite[i]
    img_refference = rad[:,:,:,i]
    img_rgb = rgb[:,:,:,i]
    ana_refference = img_refference.astype(float)
    ana_refference = Variable(torch.from_numpy(ana_refference).float())
    #image_transpose
    img_input = img_rgb.astype(float)

    h = ana_refference.shape[0]
    w = ana_refference.shape[1]
    chanel = ana_refference.shape[2]
    img_input=np.transpose(img_input, [2,0,1])
 #input_data_construction
    input = img_input
    with torch.no_grad():
        target=ana_refference.reshape(1, -1, ana_refference.shape[0], ana_refference.shape[1])
        epoch = test_ite - 1
        # model_input
        if cuda:
            print("=> use gpu id: '{}'".format(opt.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
            if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
        path = '%s%s%s' % ("checkpoint/", name, "/model_epoch_")
        type = ".pth"
        model = torch.load(path + str(epoch + 1) + type, map_location=lambda storage, loc: storage)["model"]
        # model_forward
        if cuda:
            model = model.cuda()
            ana_refference = ana_refference.cuda()
            target = target.cuda()
            mask3d_batch=mask3d_batch.cuda()
        else:
            model = model.cpu()

        input=gen_meas_torch(target[:,:,192:-192,192:-192],mask3d_batch[:,:,64:-64,64:-64])
        start_time = time.time()
        out = model(input)
        input_temp = input.cpu()

        input_temp=input_temp.data[0].numpy().astype(np.float32)
        input_temp=np.transpose(input_temp, [1, 2, 0])
        # save to .mat
        elapsed_time = time.time() - start_time
        print("It takes {}s for processing".format(elapsed_time))
        temp = out.permute(2, 3, 1, 0)

        outana = temp[:, :, :, 0]
        img_refference_save=img_refference[192:-192,192:-192,:]

        out_temp = out.cpu()
        output_temp = out_temp.data[0].numpy().astype(np.float32)
        output_temp = np.transpose(output_temp, [1, 2, 0])
        output = output_temp
        wtiff(output, output.shape[2], output.shape[0], output.shape[1],outfile+str(i)+'.tiff')
        wtiff(img_refference_save, img_refference_save.shape[2], img_refference_save.shape[0], img_refference_save.shape[1], outfile + str(i) + '_ref.tiff')
        wtiff(input_temp, input_temp.shape[2], input_temp.shape[0],
              input_temp.shape[1], outfile + str(i) + '_input.tiff')
        index[:,:,i] = evaluate.analysis_accu(ana_refference[192:-192,192:-192,:], outana, ratio)

x=index[:, 0,:].mean(dim=1)
print("\r===>Accu:    CC    PSNR   SSIM    SAM")
print("           {:.4f} {:.4f} {:.4f} {:.4f}".format(x[0].data, x[1].data,x[2].data, x[3].data))


