import argparse, os
import scipy.io as sio
import torch
import math
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import DatasetFromHdf5,TestFromHdf5
from ssim_torch import SSIM
from utils import Loss_SAM
from store2tiff import writeTiff as wtiff
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

from model import model_generator
# Training
parser = argparse.ArgumentParser(description="DL4sSR Spectral Compressive Imaging Toolbox--Mean as Input without mask")
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--method", default="CanNet", help="method name")
parser.add_argument("--accumulation-steps", type=int, default=1, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10,help="Change the learning rate, Default: n=10")
parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda?")
parser.add_argument("--resume", default=r"", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.1, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

#main fuction
opt = parser.parse_args()
print(opt)

def PSNR(img_base, img_out):
    mse = torch.mean((img_base- img_out) ** 2,0)
    mse = torch.mean(mse, 0)
    rmse = mse**0.5
    temp=torch.log(1 / rmse)/math.log(10)
    PSNR = 20 * temp
    return PSNR.mean()

def generate_masks(channel,batch_size):
    mask = sio.loadmat('mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:,:,np.newaxis],(1,1,channel))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).float()
    return mask3d_batch

def gen_meas_torch(data_batch, mask3d_batch, is_training=True):
    nC = data_batch.shape[1]
    if is_training is False:
        [batch_size, nC, H, W] = data_batch.shape
        mask3d_batch = (mask3d_batch[0,:,:,:]).expand([batch_size, nC, H, W]).cuda().float()
    temp = shift(mask3d_batch*data_batch)
    meas = torch.sum(temp, 1)/nC*2          # meas scale
    y_temp = shift_back(meas,nC)
    PhiTy = torch.mul(y_temp, mask3d_batch)
    return meas,PhiTy

def shift(inputs, step=1):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col+(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,step*i:step*i+col] = inputs[:,i,:,:]
    return output

def shift_back(inputs,nC,step=1):          # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col-(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,:] = inputs[:,:,step*i:step*i+col-(nC-1)*step]
    return output

mask3d_batch = generate_masks(31, opt.batchSize)

print("===> Loading datasets")
train_set = DatasetFromHdf5("26train_256_enhanced.h5")
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

test_set = TestFromHdf5("CAVE_6test_256.h5")
test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)


print("===> Building model")
model = model_generator(opt.method, pretrained_model_path=None)
opt_chosed = 'adam'
total_iteration = len(training_data_loader)*opt.nEpochs/opt.accumulation_steps
criterion = nn.L1Loss(reduction='mean')
# criterion = nn.MSELoss(reduction='mean')
filepath = "checkpoint/"+opt.method+"_step" + str(opt.step) + "_b" + str(
    opt.batchSize * opt.accumulation_steps) + "_" + opt_chosed + "_L1loss_adalr" + str(opt.lr) + "/"
print(filepath)

s = sum([np.prod(list(p.size())) for p in model.parameters()])
print('Number of params: %d' % s)
SSIMloss=SSIM()
SAMloss=Loss_SAM()
cuda = opt.cuda
if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
cudnn.benchmark = True

def train(training_data_loader, mask3d_batch,optimizer, model, criterion, epoch, accumulation_steps, loss_vaule,scheduler):
    # loss_avg=[]
    model.train()
    for iteration, batch in enumerate(training_data_loader, 0):
        target = Variable(batch[0]).reshape(1,31,256,256)
        starttime = time.time()
        if opt.cuda:
            target = target.cuda()
            mask3d_batch=mask3d_batch.cuda()
        # _,input=gen_meas_torch(target,mask3d_batch)

        input=torch.mean(target,dim=1)
        input=input.reshape(1,1,256,256).repeat(1,31,1,1)

        output = model(input)

        loss = criterion(output, target)
        ssim = SSIMloss(output, target)
        psnr = PSNR(output, target)
        # accumulation training
        if accumulation_steps-1:
            loss=loss/accumulation_steps
            loss.backward()
            if ((iteration + 1) % accumulation_steps) == 0:
                nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),opt.clip)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            temp=loss.data
            temp=temp.cpu()
            loss_vaule.append(temp.numpy().astype(np.float32))
            # print("===> Epoch[{}]({}/{}): HR: {:.6f},LR: {:.6f},X0: {:.6f},Final: {:.6f},Loss: {:.6f},".format(epoch, iteration, len(training_data_loader), loss_hr.data, loss_lr.data, loss_X0.data, loss_final.data, loss.data))
            endtime = time.time()
            print("\r ===> Epoch[{}]({}/{}): Loss: {:.10f} SSIM:{:.4f} PSNR:{:.4f} Time:{:.6f} lr={}".format(epoch, iteration, len(training_data_loader), loss.data,ssim.data,psnr,endtime - starttime,optimizer.param_groups[0]["lr"]),end='')

def test(test_data_loader,outfile,mask3d_batch,ssim_list,psnr_list,sam_list,epoch):
    with torch.no_grad():
        for iteration, batch in enumerate(test_data_loader, 0):
            target = Variable(batch[0]).reshape(1,31,256,256)
            starttime = time.time()
            if opt.cuda:
                target = target.cuda()
                mask3d_batch = mask3d_batch.cuda()
            # _,input=gen_meas_torch(target,mask3d_batch)
            input = torch.mean(target, dim=1)
            input =input.reshape(1, 1, 256, 256).repeat(1,31,1,1)
            # in_temp = mea.cpu()
            # in_temp = in_temp.data.numpy().astype(np.float32)
            # in_temp = np.transpose(in_temp, [1, 2, 0])
            # wtiff(in_temp, in_temp.shape[2], in_temp.shape[0], in_temp.shape[1], outfile + str(iteration + 1) + 'snap.tiff')
            # in_temp = input.cpu()
            # in_temp = in_temp.data[0].numpy().astype(np.float32)
            # in_temp = np.transpose(in_temp, [1, 2, 0])
            # wtiff(in_temp, in_temp.shape[2], in_temp.shape[0], in_temp.shape[1], outfile + str(iteration + 1) + 'input.tiff')
            # in_temp = target.cpu()
            # in_temp = in_temp.data[0].numpy().astype(np.float32)
            # in_temp = np.transpose(in_temp, [1, 2, 0])
            # wtiff(in_temp, in_temp.shape[2], in_temp.shape[0], in_temp.shape[1], outfile + str(iteration + 1) + 'target.tiff')
            output = model(input)
            endtime = time.time()
            ssim = SSIMloss(output, target)
            psnr = PSNR(output, target)
            sam = SAMloss(output, target)
            temp = sam.data
            temp = temp.cpu()
            sam_list.append(temp.numpy().astype(np.float32))
            temp = ssim.data
            temp = temp.cpu()
            ssim_list.append(temp.numpy().astype(np.float32))
            temp = psnr.cpu()
            psnr_list.append(temp.detach().numpy().astype(np.float32))
            print("\r ===> Epoch[{}]({}/{}): SSIM:{:.4f} PSNR:{:.4f} SAM:{:.4f} Time:{:.6f}".format(epoch, iteration+1,len(test_data_loader),ssim.data, psnr,sam.data,endtime-starttime),end='')
            if epoch==opt.nEpochs:
                out_temp = output.cpu()
                output_temp = out_temp.data[0].numpy().astype(np.float32)
                output_temp = np.transpose(output_temp, [1, 2, 0])
                output = output_temp
                wtiff(output, output.shape[2], output.shape[0], output.shape[1], outfile + str(iteration+1) + '.tiff')


def save_checkpoint(model, epoch,file):
    model_out_path = '%s%s%d%s' % (file, "model_epoch_",epoch,".pth")
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(file):
        os.makedirs(file)
    torch.save(state, model_out_path)
    print("\r Checkpoint saved to {}".format(model_out_path),end='')

print("===> Setting GPU")
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

    # optionally resume from a checkpoint
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
if opt.pretrained:
    if os.path.isfile(opt.pretrained):
        print("=> loading model '{}'".format(opt.pretrained))
        weights = torch.load(opt.pretrained)
        model.load_state_dict(weights['model'].state_dict())
    else:
        print("=> no model found at '{}'".format(opt.pretrained))

print("===> Setting Optimizer")
opt_Adam = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
opt_SGD = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
opt_RMSprop = optim.RMSprop(model.parameters(), lr=opt.lr, alpha=0.9, weight_decay=opt.weight_decay)
opt_dic = {'adam': opt_Adam, 'sgd': opt_SGD, 'rmsp': opt_RMSprop}
optimizer = opt_dic[opt_chosed]
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5, patience=0)

print("===> Training")
loss_show = []
outfile = 'outputnewlr_meaninput' + filepath[filepath.index('/'):]
if not os.path.exists(outfile):
    os.makedirs(outfile)
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    loss = []
    psnr_list=[]
    ssim_list = []
    sam_list=[]
    time_sta = time.time()
    train(training_data_loader, mask3d_batch,optimizer, model, criterion, epoch, opt.accumulation_steps, loss,scheduler)
    scheduler.step(np.mean(loss))
    # save_checkpoint(model, epoch, filepath)
    time_end = time.time()
    time_sta_test = time.time()
    test(test_data_loader, outfile,mask3d_batch, ssim_list, psnr_list,sam_list, epoch)
    time_end_test = time.time()
    print("===> Epoch[{}]: Loss: {:.6f} Time:{:.4f} ////  SSIM:{:.4f} PSNR:{:.4f} SAM:{:.4f} Time:{:.4f} ".format(epoch, np.mean(loss), time_end - time_sta,np.mean(ssim_list),np.mean(psnr_list),np.mean(sam_list),time_end_test - time_sta_test))
    # loss_show.append(np.mean(loss))
with open(outfile+'results.txt', 'w') as f:
    for i in range(len(ssim_list)):
        out_str=str(i+1)+'  '+str(ssim_list[i])+'  '+str(psnr_list[i])+'  '+str(sam_list[i])
        f.write(out_str+'\n')





