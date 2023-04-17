import argparse, os
import torch
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage as io
import sys
sys.path.append("..")
import evaluate
from store2tiff import writeTiff as wtiff

parser = argparse.ArgumentParser(description="PyTorch DL4sSR Demo")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda?")
parser.add_argument("--scale", default=1, type=int, help="scale factor")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--name", default="CanNet_10_b8_adam_L1loss_adalr0.0004", type=str, help="model name")
opt = parser.parse_args()
cuda = opt.cuda
name=opt.name
outfile='output/'+opt.name+'/'
if not os.path.exists(outfile):
    os.makedirs(outfile)
data = io.loadmat('cloor_42test.mat')
gray = data['test_data']

label=data['test_label_rgb']
ratio=opt.scale
bestepoch=185
band_num=label.shape[2]
ite=np.ones(label.shape[3],int)*bestepoch
index=torch.zeros((5,band_num+1,len(ite)))
for i in range(label.shape[3]):
    test_ite = ite[i]
    img_refference = label[:,:,:,i]
    img_gray = gray[:,:,:,i]
    ana_refference = img_refference.astype(float)
    ana_refference = Variable(torch.from_numpy(ana_refference).float())
    #image_transpose
    img_input = img_gray.astype(float)

    h = ana_refference.shape[0]
    w = ana_refference.shape[1]
    chanel = ana_refference.shape[2]
    img_input=np.transpose(img_input, [2,0,1])
    #input_data_construction
    input = img_input
    with torch.no_grad():
        input = Variable(torch.from_numpy(input).float()).view(1, -1, input.shape[1], input.shape[2])
        epoch = test_ite - 1
        # model_input
        path = '%s%s%s' % ("checkpoint/", name, "/model_epoch_")
        type = ".pth"
        model = torch.load(path + str(epoch + 1) + type, map_location=lambda storage, loc: storage)["model"]
        if cuda:
            print("=> use gpu id: '{}'".format(opt.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
            if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
        # model_forward
        if cuda:
            model = model.cuda()
            input = input.cuda()

            ana_refference = ana_refference.cuda()
        else:
            model = model.cpu()
        start_time = time.time()
        out = model(input[:,:,0:448,0:448])
        out_temp = out.cpu()
        # save to .mat
        elapsed_time = time.time() - start_time
        print("It takes {}s for processing".format(elapsed_time))
        temp = out.permute(2, 3, 1, 0)
        output_temp = out_temp.data[0].numpy().astype(np.float32)
        output_temp = np.transpose(output_temp, [1, 2, 0])
        output=output_temp
        outana = temp[:, :, :, 0]
        wtiff(output, output.shape[2], output.shape[0], output.shape[1],outfile+str(i)+'.tiff')
        index[:,:,i] = evaluate.analysis_accu(ana_refference[0:448,0:448,:], outana[0:448,0:448,:], ratio)
torch.set_printoptions(precision=7)
print(index[:, 0,:].mean(dim=1))

