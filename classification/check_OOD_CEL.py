'''
command for running experiments for all 10 classes - 
Assumes models saved in saved_models/class<class_num>_net.pth
python check_OOD_CEL.py --cuda --dataroot dataset --batchSize 50 --gpu 3 --n 5 --indist_class 10 --l 900

command for running experiment for a single class
python check_OOD_CEL.py --cuda --dataroot dataset --batchSize 50 --gpu 0 --n 5 --net tmp/best_net.pth --ood_dataset cifar_non3_class  --indist_class 3 --l 900 --lmbda 0
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

from dataset import CIFAR10
import PIL

from avt_wrn import Regressor as wrn_regressor

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
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--trials', type=int, default=5, help='Number of trials for cal/val split')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--net', default='', help="path to trained model")
parser.add_argument('--manualSeed', type=int, default=2535, help='manual seed')

parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# OOD detection params
parser.add_argument('--n', type=int, default=5, help='martingale parameter n')
parser.add_argument('--epsilon', type=float, default=0.92, help='power martingale parameter epsilon')
parser.add_argument('--l', type=int, default=600, help='calibration set size <= 10,000')
parser.add_argument('--ood_dataset', default='', help='use cifar_non{}_class.format(in-dist_class) for one-class OOD detection problem')

parser.add_argument('--indist_class', default=0, type=int)
parser.add_argument('--rot_bucket_width', default=10, type=int)
parser.add_argument('--m', default=0.1, type=float)
parser.add_argument('--lmbda', default=0, type=float) # SHOULD BE always 0, as testing will be only with CE loss


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



in_train_dataset, in_test_dataset, ood_test_dataset, in_train_dataloader, in_test_dataloader, out_test_dataloader, net = None, None, None, None, None, None, None

if opt.indist_class != 10:
    net = wrn_regressor().to(device) # wide resenet

    if opt.cuda:
        net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

    if opt.net != '':
        net.load_state_dict(torch.load(opt.net))

    cifar10_ood_classes = []
    for i in range(10):
        if i != opt.indist_class:
            cifar10_ood_classes.append(i)

    ood_test_dataset = CIFAR10(root=opt.dataroot, fillcolor=(128,128,128), download=True, resample=PIL.Image.BILINEAR, train=False,
                    
                    transform_pre=None,

                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]), class_list=cifar10_ood_classes, rot_bucket_width=opt.rot_bucket_width)

    in_test_dataset = CIFAR10(root=opt.dataroot, fillcolor=(128,128,128), download=True, resample=PIL.Image.BILINEAR, train=False,
                    
                        transform_pre=None,

                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]), class_list=[opt.indist_class], rot_bucket_width=opt.rot_bucket_width)


    in_train_dataset = CIFAR10(root=opt.dataroot, fillcolor=(128,128,128), download=True, resample=PIL.Image.BILINEAR, train=True,
                    
                        transform_pre=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                            ]),

                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]), class_list=[opt.indist_class], rot_bucket_width=opt.rot_bucket_width)

        

    in_train_dataloader = torch.utils.data.DataLoader(in_train_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

    in_test_dataloader = torch.utils.data.DataLoader(in_test_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

    out_test_dataloader = torch.utils.data.DataLoader(ood_test_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))


def tc_loss(zs, m, is_indist=0, means=0):
    if is_indist:
        means = zs.mean(0).unsqueeze(0)
    else:
        means = means.unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).to(device) * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0) # 2D loss shape is batchsizeX4
    return loss

def calc_CEL_TC_loss(test_dataloader, m=opt.m, lmbda=opt.lmbda, criterion=nn.CrossEntropyLoss(reduction='none'), is_indist=0, means=0):
    loss = []
    net.eval()

    counter = 0

    rot1 = torch.zeros((opt.batchSize), dtype=torch.long)
    rot2 = torch.ones((opt.batchSize), dtype=torch.long)
    rot3 = 2*torch.ones((opt.batchSize), dtype=torch.long)
    rot4 = 3*torch.ones((opt.batchSize), dtype=torch.long)
    target_rot = torch.cat((rot1, rot2, rot3, rot4))
    target_rot = target_rot.to(device)

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
        # pdb.set_trace()

        tc = tc_loss(zs=zs, m=m, is_indist=is_indist, means=means) # 2D shape of tc is batchSizeX4
        
        ce = criterion(pred_rot, target_rot) # 1D ce shape is |batchsizeX4|
        ce = ce.reshape(opt.batchSize,4)

        err = ce + lmbda * tc # 2D err shape is batchSizeX4
        err = err.mean(1) # 1D err shape is batchSize
        err = err.detach().cpu().numpy()
        
        for i in range(err.shape[0]):
            loss.append(err[i])
        
        counter += 1

    return np.array(loss)

def calc_indist_train_means(train_dataloader, m=0.1, lmbda=0.1):
    
    net.eval()
    counter = 0
    in_dist_means = None

    rot1 = torch.zeros((opt.batchSize), dtype=torch.long)
    rot2 = torch.ones((opt.batchSize), dtype=torch.long)
    rot3 = 2*torch.ones((opt.batchSize), dtype=torch.long)
    rot4 = 3*torch.ones((opt.batchSize), dtype=torch.long)
    target_rot = torch.cat((rot1, rot2, rot3, rot4))
    target_rot = target_rot.to(device)

    for _, data in enumerate(train_dataloader, 0):
        img1 = data[0].to(device)
        img2 = data[1].to(device)
        img3 = data[2].to(device)
        img4 = data[3].to(device)
        img5 = data[4].to(device)

        img_org_batch = img1.repeat(4,1,1,1)       
        img_rot_batch = torch.cat((img2,img3,img4,img5), dim=0)      

        penul_feat, _ = net(img_org_batch, img_rot_batch)

        zs = penul_feat.view(opt.batchSize,4,-1)

        if in_dist_means is None:
            in_dist_means = zs.mean(0).detach()
            in_dist_means = in_dist_means.to(device)
        else:
            in_dist_means += zs.mean(0).detach()
        
        counter += 1
    
    return in_dist_means/counter #in_dist_means shape 1X4X|penul_feat| 


# def calc_power_MV(n, epsilon, p_values, is_val):
#     p_values = np.array(p_values) # p_values = 2-dim list (nX|dataset|)
#     p_values[p_values==0]+=1e-10 
#     p_values = epsilon*(p_values**(epsilon-1))
#     cum_prod = np.cumprod(p_values,0)
#     power_martingale_values = np.max(cum_prod,0)

#     if is_val==0:
#         np.savez("{}_power_martingale_values_n{}.npz".format(opt.ood_dataset,n), mv=power_martingale_values)
#     else:
#         np.savez("cifar10_class{}_power_martingale_values_n{}.npz".format(opt.indist_class, n), mv=power_martingale_values)
#     return np.array(power_martingale_values)

def calc_power_MV(n, epsilon, p_values, is_val):
    p_values = np.array(p_values) # p_values = 2-dim list (nX|dataset|)
    p_values[p_values==0]+=1e-10 
    p_values = epsilon*(p_values**(epsilon-1))
    power_martingale_values = np.product(p_values,0)

    if is_val==0:
        np.savez("{}_power_martingale_values_n{}.npz".format(opt.ood_dataset,n), mv=power_martingale_values)
    else:
        np.savez("cifar10_class{}_power_martingale_values_n{}.npz".format(opt.indist_class, n), mv=power_martingale_values)
    return np.array(power_martingale_values)
    
def integrand(x, j, n, p_values):
        result = 1
        for i in range(n):
            result *= x*(p_values[i][j]**(x-1.0))
        return result

def calc_simple_mixture_MV(n, epsilon, p_values, is_val):
    p_values = np.array(p_values) # p_values = 2-dim list (nX|dataset|)
    p_values[p_values==0]+=1e-10 

    sm_martingale_values = []
    for j in range(p_values.shape[1]):
        sm_martingale_value, _ = integrate.quad(integrand, 0.0, 1.0, args=(j, n, p_values,))
        sm_martingale_values.append(sm_martingale_value)

    if is_val==0:
        np.savez("{}_mixture_martingale_values_n{}.npz".format(opt.ood_dataset,n), mv=sm_martingale_values)
    else:
        np.savez("cifar10_class{}_mixture_martingale_values_n{}.npz".format(opt.indist_class, n), mv=sm_martingale_values)

    return np.array(sm_martingale_values)

def calc_p_values(n, test_dataloader, test_cel, cal_set_cel, is_val_set):
    all_p_values = [] # 2-dim list (nX|dataset|) for storing p-values
    for i in range(n):
        cel = test_cel[i]

        cel_repeat = cel.reshape(1,-1)
        cal_repeat = cal_set_cel.reshape(-1,1)

        compare = cal_repeat>=cel_repeat
        all_p_values.append(np.mean(compare,0))
    
    if is_val_set==1:
        np.savez("cifar10_class{}_p_values_n{}.npz".format(opt.indist_class, n), p_values=all_p_values)
    else:
        np.savez("{}_p_values_n{}.npz".format(opt.ood_dataset,n), p_values=all_p_values)
    return all_p_values

def checkOOD_CEL():  

    n = opt.n
    ood_set_cel = []

    indist_means = calc_indist_train_means(in_train_dataloader)

    in_dist_cel_list = []
    for _ in range(n):
 
        in_dist_cel = calc_CEL_TC_loss(in_test_dataloader, is_indist=1)
        in_dist_cel_list.append(in_dist_cel)
        out_dist_cel = calc_CEL_TC_loss(out_test_dataloader, means=indist_means)       
        ood_set_cel.append(out_dist_cel)

    roc_list = [] 
    for _ in range(opt.trials):
        val_set_cel = []
        cal_set_cel = []
        indices = np.random.permutation(len(in_dist_cel_list[0]))
        for in_dist_cel in in_dist_cel_list:
            in_dist_cel = in_dist_cel[indices]
            cal_set_cel.append(in_dist_cel[opt.l:])
            val_set_cel.append(in_dist_cel[:opt.l])

        # calculate cel for CIFAR10 
        np.savez("CIFAR10_class_{}_cal_set_cel.npz".format(opt.indist_class),cal_set_cel=np.array(cal_set_cel))
        cal_set_cel = np.load("CIFAR10_class_{}_cal_set_cel.npz".format(opt.indist_class))['cal_set_cel']

        cal_set_cel = cal_set_cel.flatten()

        np.savez("CIFAR10_class{}_val_set_cel.npz".format(opt.indist_class),val_set_cel=np.array(val_set_cel)) # val_set_cel = 2D array of dim nX|val set|

        np.savez("{}_cel.npz".format(opt.ood_dataset),ood_set_cel=np.array(ood_set_cel)) # ood_set_cel = 2D array of dim nX|ood dataset|

        ood_test_cel = np.load("{}_cel.npz".format(opt.ood_dataset))['ood_set_cel']
        val_in_dist_test_cel = np.load("CIFAR10_class{}_val_set_cel.npz".format(opt.indist_class))['val_set_cel']

        ood_p_values = calc_p_values(n, out_test_dataloader, ood_test_cel, cal_set_cel, 0)

        # calculate Martingale values for OOD dataset 
        ood_martingale_values = calc_power_MV(n, epsilon=opt.epsilon, p_values=ood_p_values, is_val=0)

        # calculate p-values for validation in-dist dataset
        val_indist_p_values = calc_p_values(n, in_test_dataloader, val_in_dist_test_cel, cal_set_cel, 1)

        # calculate Martingale values for validation in-distribution set
        val_indist_martingale_values = calc_power_MV(n, epsilon=opt.epsilon, p_values=val_indist_p_values, is_val=1)

        val_indist_martingale_values = np.sort(val_indist_martingale_values)

        tau = val_indist_martingale_values[int(len(val_indist_martingale_values)*0.95)] #OOD detection threshold at 95% TPR

        tnr = np.mean(ood_martingale_values>tau)
        print("n: {}, tau: {}, TNR: {}".format(n, tau, tnr*100.))

        aur_roc =  getAUROC(opt.n)
        roc_list.append(aur_roc)

    roc_list = np.array(roc_list)
    return roc_list.mean(), roc_list.std()

def getAUROC(n):
    ood_mv = np.load("{}_power_martingale_values_n{}.npz".format(opt.ood_dataset,n))['mv']
    indist_mv = np.load("cifar10_class{}_power_martingale_values_n{}.npz".format(opt.indist_class, n))['mv']
    mv = np.concatenate((indist_mv, ood_mv))

    indist_label = np.zeros((opt.l))
    ood_label = np.ones((ood_test_dataset.__len__()))
    label = np.concatenate((indist_label, ood_label))

    from sklearn.metrics import roc_auc_score
    au_roc = roc_auc_score(label, mv)*100
    return au_roc

def runAllExperiments():
    global in_train_dataloader, in_test_dataloader, out_test_dataloader, ood_test_dataset, in_test_dataset, in_train_dataset, net
    all_results = []
    for i in range(10):
        opt.indist_class = i
        opt.net = "saved_models/class{}_net.pth".format(i)
        opt.ood_dataset =  "cifar_non{}_class".format(i)
        print(opt)

        net = wrn_regressor().to(device) # wide resenet

        if opt.cuda:
            net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

        if opt.net != '':
            net.load_state_dict(torch.load(opt.net))

        cifar10_ood_classes = []
        for j in range(10):
            if j != opt.indist_class:
                cifar10_ood_classes.append(j)

        ood_test_dataset = CIFAR10(root=opt.dataroot, fillcolor=(128,128,128), download=True, resample=PIL.Image.BILINEAR, train=False,
                        
                        transform_pre=None,

                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]), class_list=cifar10_ood_classes, rot_bucket_width=opt.rot_bucket_width)

        in_test_dataset = CIFAR10(root=opt.dataroot, fillcolor=(128,128,128), download=True, resample=PIL.Image.BILINEAR, train=False,
                        
                            transform_pre=None,

                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]), class_list=[opt.indist_class], rot_bucket_width=opt.rot_bucket_width)


        in_train_dataset = CIFAR10(root=opt.dataroot, fillcolor=(128,128,128), download=True, resample=PIL.Image.BILINEAR, train=True,
                        
                            transform_pre=transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                ]),

                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]), class_list=[opt.indist_class], rot_bucket_width=opt.rot_bucket_width)

        in_train_dataloader = torch.utils.data.DataLoader(in_train_dataset, batch_size=opt.batchSize,
                                                shuffle=False, num_workers=int(opt.workers))

        in_test_dataloader = torch.utils.data.DataLoader(in_test_dataset, batch_size=opt.batchSize,
                                                shuffle=False, num_workers=int(opt.workers))

        out_test_dataloader = torch.utils.data.DataLoader(ood_test_dataset, batch_size=opt.batchSize,
                                                shuffle=False, num_workers=int(opt.workers))

        avr_roc,std_roc = checkOOD_CEL()
        print("Average ROC and std for class {}: [{}] [{}]".format(i, avr_roc, std_roc))

        all_results.append("{} {}".format(avr_roc,std_roc))

    with open("all_results_valsize_{}.txt".format(opt.l),'w') as f: f.write("\n".join(all_results))


if __name__ == "__main__":
    if opt.indist_class == 10:
        runAllExperiments() # for running experiments for all 10 classes
    else: # for running for a specific class
        avr_roc,std_roc = checkOOD_CEL()
        print("Average ROC and std: [{}] [{}]".format(avr_roc, std_roc))
