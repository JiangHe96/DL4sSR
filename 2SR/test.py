
import argparse, os
import torch
from torch.autograd import Variable
import time
import numpy as np
from torch.utils.data import DataLoader
import hdf5storage
from hsi_dataset import TrainDataset, ValidDataset
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR,Loss_SAM

import sys
sys.path.append("..")
from model import model_generator
def saveCube(path, cube, bands=np.linspace(400,700,num=31), norm_factor=None):
    hdf5storage.write({u'cube': cube,
                       u'bands': bands,
                       u'norm_factor': norm_factor}, '.',
                       path, matlab_compatible=True)

# model_input
parser = argparse.ArgumentParser(description="DL4sSR Spectral Recovery Demo")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda?")
parser.add_argument("--epoch", default=300, type=int, help="scale factor")
parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 0)")
parser.add_argument("--model", default='CanNet', type=str, help="model name")
parser.add_argument("--chk_file", default='CanNet_b8_adalr0.0008', type=str, help="model path")
parser.add_argument("--time", default='2023_04_17_23_09_36', type=str, help="model path")
parser.add_argument("--data_root", type=str, default='dataset/')
opt = parser.parse_args()
cuda = opt.cuda
epoch=opt.epoch
filename=r'checkpoint/'
outname=r'output/'
eopchname='%s%d%s' %('/net_',epoch,'epoch.pth')

model = model_generator(opt.model)
model.load_state_dict(torch.load('%s%s%s%s%s' %(filename,opt.chk_file,'/',opt.time,eopchname))["state_dict"])
model=model.cuda()
if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
with torch.no_grad():
    criterion_mrae = Loss_MRAE()
    criterion_rmse = Loss_RMSE()
    criterion_psnr = Loss_PSNR()
    criterion_sam= Loss_SAM()
    val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_sam = AverageMeter()
    timelist=[]
    for i, (input, target) in enumerate(val_loader):
        input = input[:,:,:480,:].cuda()
        target = target[:,:,:480,:].cuda()
        # compute output
        starttime = time.time()
        output = model(input)
        endtime = time.time()
        timelist.append(endtime-starttime)

        out_temp = output.cpu()
        out = out_temp[0, :, :, :].permute(1, 2, 0).numpy().astype(np.float32)
        loss_mrae = criterion_mrae(output, target)
        loss_rmse = criterion_rmse(output, target)
        loss_psnr = criterion_psnr(output, target)
        loss_sam = criterion_sam(output, target)

        outfile='%s%s%s%s' %(outname,opt.chk_file,'/',opt.time)
        isExists = os.path.exists(outfile)
        if not isExists:
            os.makedirs(outfile)
            print(outfile + ' sucess')

        saveCube('%s%s%d%s' %(outfile,'/ARAD_1K_', i + 901, '.mat'), out)
        # record loss
        if ~torch.isinf(loss_mrae.data):
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            losses_sam.update(loss_sam.data)
print("Time Used: ",np.mean(timelist))
print("===>Accuracy:    MRAE     RMSE      PSNR     SAM")
print("                {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(losses_mrae.avg.item(),losses_rmse.avg.item(),losses_psnr.avg.item(),losses_sam.avg.item()))


