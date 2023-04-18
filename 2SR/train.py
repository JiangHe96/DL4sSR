import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset import TrainDataset, ValidDataset
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR,Loss_SAM
import datetime
import time
import sys
sys.path.append("..")
from model import model_generator

parser = argparse.ArgumentParser(description="DL4sSR Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='SSDCN')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--accumulation-steps", type=int, default=8, help="Training batch size")
parser.add_argument("--threads", type=int, default=0, help="threads")
parser.add_argument("--start_epoch", type=int, default=1, help="start of epochs")
parser.add_argument("--end_epoch", type=int, default=200, help="number of epochs")
parser.add_argument("--save_epoch", type=int, default=5, help="save of epochs")
parser.add_argument("--lr", type=float, default=0.0008, help="initial learning rate")
parser.add_argument("--outf", type=str, default='checkpoint/', help='path log files')
parser.add_argument("--data_root", type=str, default='dataset/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpus", type=str, default='0', help='gpu name')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

# model
model = model_generator(opt.method, opt.pretrained_model_path)
filepath=opt.method + "_b" + str(opt.batch_size*opt.accumulation_steps) +  "_adalr" + str(opt.lr)+'/'
print(filepath)
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# load dataset
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)

val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = 5000
total_iteration = per_epoch_iteration*opt.end_epoch
print(f"Iteration per epoch: {per_epoch_iteration}")

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_sam= Loss_SAM()
criterion_l1 = nn.L1Loss(reduction='mean')

# output path
file_log=opt.outf
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf+filepath + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
print(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    criterion_l1.cuda()
    print('GPU {} is used!'.format(opt.gpus))

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-8)

# logging
log_dir = os.path.join(file_log, opt.method + "_b" + str(opt.batch_size*opt.accumulation_steps) +  "_l1loss_adalr" + str(opt.lr)+'.log')
logger = initialize_logger(log_dir)
logger.info(" Batchsize[%03d], Patchsize[%03d], Stride[%03d]" % (opt.batch_size*opt.accumulation_steps, opt.patch_size, opt.stride))

# Resume
resume_file = opt.pretrained_model_path
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    cudnn.benchmark = False
    record_mrae_loss = 1000
    epoch=0
    iteration = 0
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=opt.threads, pin_memory=True)
    while iteration < total_iteration:
        epoch = epoch + 1
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads,
                                  pin_memory=True, drop_last=True)

        starttime = time.time()
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            # optimizer.zero_grad()
            output = model(images)
            loss = criterion_l1(output, labels)
            if opt.accumulation_steps - 1:
                # accumulation training
                loss = loss / opt.accumulation_steps
                loss.backward()
                if ((iteration + 1) % opt.accumulation_steps) == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            losses.update(loss.data)
            iteration = iteration+1
            if iteration % 20 == 0:
                endtime = time.time()
                print("\r ===> Epoch[{}]({}/{}): train_losses.avg={:.6f} Time:{:.6f}s lr={:.9f}".format(epoch, iteration%per_epoch_iteration,per_epoch_iteration,losses.avg,endtime - starttime,lr),end='')
                starttime = time.time()

            if iteration % per_epoch_iteration == 0:
                break
        mrae_loss, rmse_loss, psnr_loss,sam_loss = validate(val_loader, model)
        print("\n MRAE {:.6f}, RMSE {:.6f}, PSNR {:.6f}, SAM {:.6f}".format(mrae_loss, rmse_loss,psnr_loss,sam_loss))
        # Save model
        if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or epoch % opt.save_epoch == 0:
            save_checkpoint(opt.outf, epoch, iteration, model, optimizer)
            if mrae_loss < record_mrae_loss:
                record_mrae_loss = mrae_loss
        # print loss
        logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Validation: %.9f, %.9f, %.9f, %.9f " % (
                    iteration, epoch, optimizer.param_groups[0]['lr'], losses.avg, mrae_loss, rmse_loss, psnr_loss,sam_loss))
    return 0

# Validate
def validate(val_loader, model):
    with torch.no_grad():
        model.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_sam = AverageMeter()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input[:,:,:480,:])
            loss_mrae = criterion_mrae(output, target[:,:,:480,:])
            loss_rmse = criterion_rmse(output, target[:,:,:480,:])
            loss_psnr = criterion_psnr(output, target[:,:,:480,:])
            loss_sam = criterion_sam(output, target[:,:,:480,:])
            # record loss
            if ~torch.isinf(loss_mrae.data):
                losses_mrae.update(loss_mrae.data)
                losses_rmse.update(loss_rmse.data)
                losses_psnr.update(loss_psnr.data)
                losses_sam.update(loss_sam.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg,losses_sam.avg

if __name__ == '__main__':
    main()
    filepath = opt.method + "_b" + str(opt.batch_size) +  "_adalr" + str(opt.lr)
    print(filepath)