import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import DatasetFromHdf5
import time
import numpy as np
import sys
sys.path.append("..")

from model import model_generator

# Training
parser = argparse.ArgumentParser(description="DL4sSR Colorization Toolbox")
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--accumulation-steps", type=int, default=1, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--method", default="SSDCN", help="method name")
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
cuda = opt.cuda
if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
print("===> Loading datasets")
train_set = DatasetFromHdf5("color_train.h5")
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print("===> Building model")
model =model_generator(opt.method, pretrained_model_path=None)
opt_chosed = 'adam'
total_iteration = len(training_data_loader)*opt.nEpochs/opt.accumulation_steps
criterion = nn.L1Loss(reduction='mean')
# criterion = nn.MSELoss(reduction='mean')
filepath = "checkpoint/"+opt.method+"_" + str(opt.step) + "_b" + str(
    opt.batchSize * opt.accumulation_steps) + "_" + opt_chosed + "_L1loss_adalr" + str(opt.lr) + "/"
print(filepath)
s = sum([np.prod(list(p.size())) for p in model.parameters()])
print('Number of params: %d' % s)

opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
cudnn.benchmark = True

def train(training_data_loader, optimizer, model, criterion, epoch, accumulation_steps, loss_vaule,scheduler):

    # loss_avg=[]
    model.train()
    for iteration, batch in enumerate(training_data_loader, 0):
        input, target = Variable(batch[0]), Variable(batch[1])
        starttime = time.time()
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        output=model(input)
        loss = criterion(output, target)

        if accumulation_steps-1:
            # accumulation training
            loss=loss/accumulation_steps
            loss.backward()
            if ((iteration + 1) % accumulation_steps) == 0:
                nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),opt.clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        temp=loss.data
        temp=temp.cpu()
        loss_vaule.append(temp.numpy().astype(np.float32))
        endtime = time.time()
        print("\r ===> Epoch[{}]({}/{}): Loss: {:.10f} Time:{:.6f} lr={}".format(epoch, iteration, len(training_data_loader), loss.data,endtime - starttime,optimizer.param_groups[0]["lr"]),end='')

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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

print("===> Training")
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    loss = []
    time_sta = time.time()
    train(training_data_loader, optimizer, model, criterion, epoch, opt.accumulation_steps, loss,scheduler)
    save_checkpoint(model, epoch, filepath)
    time_end = time.time()
    print("===> Epoch[{}]: Loss: {:.10f} Time:{:.6f}".format(epoch, np.mean(loss), time_end - time_sta))