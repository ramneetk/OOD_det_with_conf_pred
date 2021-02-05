# command to run python main.py --cuda --outf $output dir$ --dataroot dataset --gpu $0/1/2/3$ --class_num $0-9$ (for class-wise training)

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

from avt_wrn import Regressor as wrn_regressor

from dataset import CIFAR10
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
parser.add_argument('--batchSize', type=int, default=50, help='input batch size as a factor of 5000')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=4500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--wgtDecay', default=5e-4, type=float)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--optimizer', default='', help="path to optimizer (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=2535, help='manual seed')
parser.add_argument('--divide', type=float, default=1000.)
parser.add_argument('--debug', default=False, type=str2bool, nargs='?', const=True, help='will call pdb.set_trace() for debugging')

parser.add_argument('--scheduler', default=False, type=str2bool, nargs='?', const=True, help='scheduler for training')
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--class_num', default=-1, type=int, help='CIFAR10 class number for training 1-class model, -1 is for complete dataset')
parser.add_argument('--m', default=0.1, type=float)
parser.add_argument('--lmbda', default=0.1, type=float)
parser.add_argument('--reg', default=False, type=str2bool, nargs='?', const=True)
parser.add_argument('--reg_param', default=10., type=float)
parser.add_argument('--rot_bucket_width', default=10, type=int)

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

def getClassWiseDataset(class_num, root, train, download, fillcolor, resample, transform_pre, transform):
    cifar10_data = CIFAR10(root=root, download=download, train=train, fillcolor=fillcolor, resample=resample, transform_pre=transform_pre, transform=transform, rot_bucket_width=opt.rot_bucket_width)
    if class_num == -1:
        return cifar10_data
    else:
        return [cifar10_data.__getitem__(i) for i in range(cifar10_data.__len__()) if cifar10_data.__getitem__(i)[-1]==class_num]

train_dataset = getClassWiseDataset(class_num=opt.class_num, root=opt.dataroot, train=True, download=True, fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,
                           transform_pre=transforms.Compose([
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                           ]),
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))

test_dataset = getClassWiseDataset(class_num=opt.class_num, root=opt.dataroot, train=False, download=True,
fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,
                           transform_pre=None,

                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))

assert train_dataset
assert test_dataset
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

print("device: ", device)

net = wrn_regressor().to(device) # wide resenet

if opt.cuda:
    net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.wgtDecay)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        opt.niter * len(train_dataloader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / opt.lr))

def calculateTotalParameters(model):
    count = 0
    for p in model.parameters():
        if p.requires_grad:
            count+=p.numel()
    
    return count

def tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).to(device) * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss


def evaluate(test_dataloader):
    
    net.eval()

    total_err = 0
    total_ce = 0
    total_tc = 0
    counter = 0

    rot1 = torch.zeros((opt.batchSize), dtype=torch.long)
    rot2 = torch.ones((opt.batchSize), dtype=torch.long)
    rot3 = 2*torch.ones((opt.batchSize), dtype=torch.long)
    rot4 = 3*torch.ones((opt.batchSize), dtype=torch.long)
    target_rot = torch.cat((rot1, rot2, rot3, rot4))
    target_rot = target_rot.to(device)


    with torch.no_grad():

        for _, data in enumerate(test_dataloader, 0):

            img1 = data[0].to(device)
            img2 = data[1].to(device)
            img3 = data[2].to(device)
            img4 = data[3].to(device)
            img5 = data[4].to(device)

            img_org_batch = img1.repeat(4,1,1,1)       
            img_rot_batch = torch.cat((img2,img3,img4,img5), dim=0)      

            penul_feat, pred_rot = net(img_org_batch, img_rot_batch)

            zs = penul_feat.view(opt.batchSize,4,-1)

            tc = tc_loss(zs=zs, m=opt.m)
            ce = criterion(pred_rot, target_rot)

            if opt.reg:
                loss = ce + opt.lmbda * tc + opt.reg_param *(zs*zs).mean()
            else:
                loss = ce + opt.lmbda * tc

            total_err += loss.item()
            total_ce += ce.item()
            total_tc += tc.item()
            counter += 1

    # print("ce loss: {}, tc loss: {}".format(total_ce/counter, total_tc/counter))
    return total_err/counter  

def train():
    best_test_err = float('inf')

    if opt.debug:
        pdb.set_trace()

    rot1 = torch.zeros((opt.batchSize), dtype=torch.long)
    rot2 = torch.ones((opt.batchSize), dtype=torch.long)
    rot3 = 2*torch.ones((opt.batchSize), dtype=torch.long)
    rot4 = 3*torch.ones((opt.batchSize), dtype=torch.long)
    target_rot = torch.cat((rot1, rot2, rot3, rot4))
    target_rot = target_rot.to(device)

    for epoch in range(opt.niter):
        net.train() 
        total_train_loss = 0
        for i, data in enumerate(train_dataloader, 0):
            if opt.scheduler:
                scheduler.step()
            
            net.zero_grad()

            img1 = data[0].to(device)
            img2 = data[1].to(device)
            img3 = data[2].to(device)
            img4 = data[3].to(device)
            img5 = data[4].to(device)

            img_org_batch = img1.repeat(4,1,1,1)       
            img_rot_batch = torch.cat((img2,img3,img4,img5), dim=0)      

            penul_feat, pred_rot = net(img_org_batch, img_rot_batch)

            zs = penul_feat.view(opt.batchSize,4,-1)

            tc = tc_loss(zs=zs, m=opt.m)
            ce = criterion(pred_rot, target_rot)

            if opt.reg:
                loss = ce + opt.lmbda * tc + opt.reg_param *(zs*zs).mean()
            else:
                loss = ce + opt.lmbda * tc
            
            total_train_loss+=loss.item()
            loss.backward()
            optimizer.step() 

        # print("For training data: ")
        train_err = evaluate(train_dataloader)    
        # print("For test data: ")
        test_err  = evaluate(test_dataloader) 

        if best_test_err > test_err:
            torch.save(net.state_dict(), '%s/best_net.pth' % (opt.outf))
            torch.save(optimizer.state_dict(), '%s/best_optimizer.pth' % (opt.outf))  
            best_test_err = test_err  

        print("Epoch: {},train loss: {} train err: {} test err: {} best test err: {}".format(epoch,total_train_loss/len(train_dataloader), train_err, test_err, best_test_err))

        # do checkpointing
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.outf, epoch+1))
            torch.save(optimizer.state_dict(), '%s/optimizer_epoch_%d.pth' % (opt.outf, epoch+1))

def test_saved_model():
    test_err  = evaluate(test_dataloader)        
    print("test err: {}".format(test_err))

if __name__ == "__main__":
    train()