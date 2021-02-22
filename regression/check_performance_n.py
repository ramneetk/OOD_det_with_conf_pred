'''
command to run - 
python check_performance_n.py --cuda --dataroot dataset --batchSize 100 --gpu 0 --net saved_models/cifar10.pth --n 50 --ood_dataset $SVHN/LSUN/CIFAR100/Places365$ --l 9000
'''

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np

from avt_resnet import Regressor as Regressor_resnet

from dataset import CIFAR10
from ood_dataset import CIFAR_OOD
import PIL


import pdb

from scipy.integrate import quad_vec
from scipy import integrate


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

parser.add_argument('--trials', type=int, default=5, help='Number of trials for cal/val split')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--net', default='', help="path load the trained network")
parser.add_argument('--manualSeed', type=int, default=2535, help='manual seed')
parser.add_argument('--shift', type=float, default=4)
parser.add_argument('--shrink', type=float, default=0.8)
parser.add_argument('--enlarge', type=float, default=1.2)

parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# OOD detection params
parser.add_argument('--n', type=int, default=5, help='no. of transformations')
parser.add_argument('--l', type=int, default=9000, help='l is the size of the validation set')
parser.add_argument('--ood_dataset', default='SVHN')

opt = parser.parse_args()
print(opt)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(opt.gpu) if use_cuda else "cpu")

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

net = Regressor_resnet(resnet_type=34).to(device)

if opt.cuda:
    net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

ood_test_dataset = CIFAR_OOD(dataset_name=opt.ood_dataset, shift=opt.shift, scale=(opt.shrink, opt.enlarge), fillcolor=(128,128,128), download=True, train=False, resample=PIL.Image.BILINEAR,
                    matrix_transform=transforms.Compose([
                        transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                    ]),

                    transform_pre=None,

                    transform=transforms.Compose([
                        transforms.Resize((32,32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]))

in_test_dataset = CIFAR10(root=opt.dataroot, shift=opt.shift, scale=(opt.shrink, opt.enlarge), fillcolor=(128,128,128), download=True, train=False, resample=PIL.Image.BILINEAR,
                        matrix_transform=transforms.Compose([
                            transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                        ]),
                        transform_pre=None,

                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))
in_test_dataloader = torch.utils.data.DataLoader(in_test_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

out_test_dataloader = torch.utils.data.DataLoader(ood_test_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

criterion = nn.MSELoss()

def calc_MSE(test_dataloader):
    mse = []
    net.eval()
    for _, data in enumerate(test_dataloader, 0):
        img1 = data[0].to(device)
        img2 = data[1].to(device)
        matrix = data[2].to(device)
        matrix = matrix.view(-1, 8)
        
        _, _, output_mu, _ = net(img1, img2)
        
        for j in range(output_mu.shape[0]):
            mse.append(criterion(output_mu[j], matrix[j]).item())

    return np.array(mse)

def calc_p_values(n, test_mse, cal_set_mse, is_val_set):

    cal_set_mse_reshaped = cal_set_mse
    cal_set_mse_reshaped = cal_set_mse_reshaped.reshape(1,-1) # cal_set_mse reshaped into row vector

    test_mse_reshaped = test_mse
    test_mse_reshaped = test_mse_reshaped.reshape(-1,1) # test_mse reshaped into column vector

    compare = test_mse_reshaped<=cal_set_mse_reshaped
    p_values = np.sum(compare, axis=1)
    p_values = p_values/len(cal_set_mse)
    
    if is_val_set==1:
        np.savez("cifar10_p_values_n{}.npz".format(n), p_values=p_values)
    else:
        np.savez("{}_p_values_n{}.npz".format(opt.ood_dataset,n), p_values=p_values)

    return p_values

def checkOOD(n = opt.n):  

    ood_set_mse = []
    in_dist_mse_list = []

    for _ in range(n):
        in_dist_mse = calc_MSE(in_test_dataloader)
        in_dist_mse_list.append(in_dist_mse)

        out_dist_mse = calc_MSE(out_test_dataloader)
        ood_set_mse.append(out_dist_mse)

    np.savez("CIFAR10_test_set_MSE.npz",test_set_mse=np.array(in_dist_mse_list))

    ood_set_mse = np.array(ood_set_mse)
    ood_set_mse = np.transpose(ood_set_mse)

    final_roc = []
    final_tnr = []
    temp = n
    for iter in range(1,temp+1):
        print("iter: ", iter)
        roc_list = []
        tnr_list = []
        for _ in range(opt.trials):
            val_set_mse = []
            cal_set_mse = []
            indices = np.random.permutation(len(in_dist_mse_list[0]))
            for in_dist_mse in in_dist_mse_list:
                in_dist_mse = in_dist_mse[indices]
                cal_set_mse.append(in_dist_mse[opt.l:])
                val_set_mse.append(in_dist_mse[:opt.l])
            # calculate MSE for CIFAR10 
            np.savez("CIFAR10_cal_set_MSE.npz",cal_set_mse=np.array(cal_set_mse))

            # cal_set_mse = cal_set_mse.flatten()
            ########## STEP 1 = for each data point, create n alphas = A(data point) #################
            cal_set_mse = np.array(cal_set_mse) # cal_set_mse = n X 9000-opt.l 
            cal_set_mse = np.transpose(cal_set_mse) # cal_set_mse = 9000-opt.l X n -> for each data point in cal set, n alphas (or cross-entropy loss cel), repeat the same for val_set_mse, for ood_set_mse this is done before looping on trials
            val_set_mse = np.array(val_set_mse)
            val_set_mse = np.transpose(val_set_mse)

            cal_set_mse = cal_set_mse[:,:iter]
            val_set_mse = val_set_mse[:,:iter]
            ood_set_mse_tmp = ood_set_mse[:,:iter]

            np.savez("CIFAR10_val_set_MSE.npz",val_set_mse=val_set_mse) # val_set_mse = 2D array of dim |val set| X n

            np.savez("{}_MSE.npz".format(opt.ood_dataset),ood_set_mse=ood_set_mse_tmp) # ood_set_mse = 2D array of dim |val set| X n

            ######## STEP 2 =  Apply F on A(data point), F(t) = summation of all the values in t 
            f_cal_set = np.sum(cal_set_mse, axis = 1)
            f_val_set = np.sum(val_set_mse, axis = 1)
            f_ood_set = np.sum(ood_set_mse_tmp, axis = 1)


            ######## STEP 3 = Calculate p-values for OOD and validation set #########
            ood_p_values = calc_p_values(iter, f_ood_set, f_cal_set, is_val_set=0)
            # calculate p-values for validation in-dist dataset
            val_indist_p_values = calc_p_values(iter, f_val_set, f_cal_set, is_val_set=1)

            val_indist_p_values = np.sort(val_indist_p_values)

            tau = val_indist_p_values[int(len(val_indist_p_values)*0.05)] #OOD detection threshold at 95% TPR

            tnr = np.mean(ood_p_values<tau)

            aur_roc =  getAUROC(iter)
            roc_list.append(aur_roc)
            tnr_list.append(tnr)

        roc_list = np.array(roc_list)
        tnr_list = np.array(tnr_list)
    
        final_roc.append(roc_list.mean())
        final_tnr.append(tnr_list.mean()*100)
        print("n: {} AVG_TNR: {}, AVG_AUROC: {}".format(iter, tnr_list.mean()*100, roc_list.mean()))

    return final_roc, final_tnr


def getAUROC(n):
    ood_p_values = np.load("{}_p_values_n{}.npz".format(opt.ood_dataset,n))['p_values']
    indist_p_values = np.load("cifar10_p_values_n{}.npz".format(n))['p_values']
    p_values = np.concatenate((indist_p_values, ood_p_values))

    indist_label = np.ones((opt.l))
    ood_label = np.zeros((ood_test_dataset.__len__()))
    label = np.concatenate((indist_label, ood_label))

    from sklearn.metrics import roc_auc_score
    au_roc = roc_auc_score(label, p_values)*100
    return au_roc

def test_diff_n():
    final_roc, final_tnr = checkOOD()

    np.savez("CIFAR10_{}_avr_roc_diff_n_{}.npz".format(opt.ood_dataset, opt.n), avr_roc=np.array(final_roc))
    np.savez("CIFAR10_{}_avr_tnr_diff_n_{}.npz".format(opt.ood_dataset, opt.n), avr_tnr=np.array(final_tnr))
 
if __name__ == "__main__":
    test_diff_n()
