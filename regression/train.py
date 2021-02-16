'''
command to train CIFAR10 dataset on Resnet34
python train.py --cuda --outf $output_dir$ --gpu 0 --dataroot dataset --batchSize 100 --archi resnet34

command to train class-wise models of CIFAR10 on WideResNet
python train.py --cuda --outf $output dir$ --dataroot dataset --archi wideresnet --batchSize 100 --gpu 0 --class_num $0-9$ --wgtDecay 0.0055
'''

from __future__ import print_function
import argparse
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

from avt_resnet import Regressor as Regressor_resnet
from avt_wrn import Regressor as Regressor_wideresnet


from baseline_avt_resnet import Regressor as Regressor_baseline

from dataset import CIFAR10
from ood_dataset import CIFAR_OOD
import PIL

import pdb

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='dataset', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=4500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--wgtDecay', default=5e-4, type=float)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--optimizer', default='', help="path to optimizer (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=2535, help='manual seed')
parser.add_argument('--shift', type=float, default=4)
parser.add_argument('--shrink', type=float, default=0.8)
parser.add_argument('--enlarge', type=float, default=1.2)
parser.add_argument('--lrMul', type=float, default=10.)
parser.add_argument('--divide', type=float, default=1000.)
parser.add_argument('--dropout', type=float, default=None)
parser.add_argument('--archi', type=str, required=True, choices=['resnet34','wideresnet', 'baseline'])
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--class_num', default=-1, type=int, help='CIFAR10 class number for training 1-class model, -1 is for complete dataset')

opt = parser.parse_args()
print(opt)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(opt.gpu) if use_cuda else "cpu")

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

def getClassWiseDataset(class_num, root, shift, scale, fillcolor, train, download, resample, matrix_transform, transform_pre, transform):
    cifar10_data = CIFAR10(root=root, shift=shift, scale=scale, fillcolor=fillcolor, download=download, train=train, resample=resample, matrix_transform=matrix_transform, transform_pre=transform_pre, transform=transform)
    if class_num == -1: # for complete CIFAR10 dataset
        return cifar10_data
    else: # for class-wise dataset
        return [cifar10_data.__getitem__(i) for i in range(cifar10_data.__len__()) if cifar10_data.__getitem__(i)[-1]==class_num]

train_dataset = getClassWiseDataset(class_num=opt.class_num, root=opt.dataroot, shift=opt.shift, scale=(opt.shrink, opt.enlarge), fillcolor=(128,128,128), train=True, download=True, resample=PIL.Image.BILINEAR,
                           matrix_transform=transforms.Compose([
                               transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                           ]),
                           transform_pre=transforms.Compose([
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                           ]),
                        #    transform_pre = None,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))

test_dataset = getClassWiseDataset(class_num=opt.class_num, root=opt.dataroot, shift=opt.shift, scale=(opt.shrink, opt.enlarge), fillcolor=(128,128,128), train=False, download=True, resample=PIL.Image.BILINEAR,
                           matrix_transform=transforms.Compose([
                               transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                           ]),

                           transform_pre=None,

                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))

assert train_dataset
assert test_dataset

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

print("train_dataset_len ", train_dataset.__len__())
print("test dataset_len ", test_dataset.__len__())
print("device: ", device)

if opt.archi=='resnet34':
    net = Regressor_resnet(resnet_type=34,dropout=opt.dropout).to(device) #resnet34
elif opt.archi=='wideresnet':
    net = Regressor_wideresnet(dropout=opt.dropout).to(device) #wide resenet
elif opt.archi=='baseline':
    net = Regressor_baseline(resnet_type=34,dropout=opt.dropout).to(device) # baseline avt_resnet34 model
else:
    raise Exception('Wrong algorithm type')

if opt.cuda:
    net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

criterion = nn.MSELoss()

# setup optimizer
fc2_params = list(map(id, net.module.fc2.parameters()))
base_params = filter(lambda p: id(p) not in fc2_params, net.parameters())

optimizer = optim.SGD([{'params':base_params}, {'params':net.module.fc2.parameters(), 'lr': opt.lr*opt.lrMul}], lr=opt.lr, momentum=0.9, weight_decay=opt.wgtDecay, nesterov=True)
if opt.optimizer != '':
    optimizer.load_state_dict(torch.load(opt.optimizer))

def calculateTotalParameters(model):
    count = 0
    for p in model.parameters():
        if p.requires_grad:
            count+=p.numel()
    
    return count


def evaluate(test_dataloader):
    # net.eval()

    total_err = 0
    counter = 0
    with torch.no_grad():

        for i, data in enumerate(test_dataloader, 0):

            img1 = data[0].to(device)
            img2 = data[1].to(device)
            matrix = data[2].to(device)
            matrix = matrix.view(-1, 8)
            
            batch_size = img1.size(0)
            f1, f2, output_mu, output_logvar = net(img1, img2)
            output_logvar = output_logvar / opt.divide
            std_sqr = torch.exp(output_logvar)
            
            err_matrix = criterion(output_mu, matrix)
            total_err += err_matrix.item()
            counter += 1

    return total_err/counter

def train():
    print("Total number of trainable parameters: {}".format(calculateTotalParameters(net)))
    best_test_err = float('inf')
    
    for epoch in range(opt.niter):
        net.train() 
        total_train_loss = 0
        for i, data in enumerate(train_dataloader, 0):
            net.zero_grad()
            img1 = data[0].to(device)
            img2 = data[1].to(device)
            matrix = data[2].to(device)
            matrix = matrix.view(-1, 8)
            
            batch_size = img1.size(0)
            f1, f2, output_mu, output_logvar = net(img1, img2)
            output_logvar = output_logvar / opt.divide
            std_sqr = torch.exp(output_logvar)
            
            err_matrix = criterion(output_mu, matrix)
            err_reg = torch.sum(output_logvar)/batch_size
            err_mse = torch.sum((output_mu - matrix)*(output_mu - matrix) / (std_sqr + 1e-4))/batch_size
            err = (torch.sum(output_logvar) + \
                   torch.sum((output_mu - matrix)*(output_mu - matrix) / (std_sqr + 1e-4))) / batch_size 
            
            total_train_loss+=err.item()
            err.backward()
            optimizer.step()
            
        train_err = evaluate(train_dataloader)    
        test_err  = evaluate(test_dataloader) 

        if best_test_err > test_err:
            torch.save(net.state_dict(), '%s/best_net.pth' % (opt.outf))
            torch.save(optimizer.state_dict(), '%s/best_optimizer.pth' % (opt.outf))  
            best_test_err = test_err  

        print("Epoch: {},train loss: {} train err: {} test err: {} best test err: {}".format(epoch,total_train_loss/len(train_dataloader), train_err, test_err, best_test_err))

        # do checkpointing
        if epoch % 100 == 49:
            torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(optimizer.state_dict(), '%s/optimizer_epoch_%d.pth' % (opt.outf, epoch))

def test_saved_model():
    test_err  = evaluate(test_dataloader)        
    print("test err: {}".format(test_err))

if __name__ == "__main__":
    train()