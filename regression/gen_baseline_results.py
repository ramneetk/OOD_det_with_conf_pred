'''
Run this for generating the required npz files
python check_OOD.py --cuda --dataroot dataset --batchSize 100 --gpu 0 --net saved_models/cifar10.pth --n 1 --ood_dataset CFIAR100 --archi resnet34 --one_class_det 0 --l 9000
'''

import numpy as np
in_mse = np.load("CIFAR10_class_-1_test_set_MSE.npz")['test_set_mse']
ood_mse = np.load("CIFAR100_MSE.npz")['ood_set_mse']

in_mse = np.transpose(in_mse)
in_mse = in_mse[:,0]
ood_mse = ood_mse[:,0]
in_mse = np.sort(in_mse)
tau = in_mse[int(0.95*len(in_mse))]
mse_tnr = 100*len(ood_mse[ood_mse>tau])/len(ood_mse)

mse = np.concatenate((in_mse, ood_mse))
indist_label = np.zeros((len(in_mse)))
ood_label = np.ones((len(ood_mse)))
label = np.concatenate((indist_label, ood_label))

from sklearn.metrics import roc_auc_score
au_roc = roc_auc_score(label, mse)*100

print("TNR: ", mse_tnr)
print("AUROC: ", roc_auc_score(label, mse)*100)