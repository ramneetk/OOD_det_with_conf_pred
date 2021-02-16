### Downloading CIFAR10 dataset 
   Set download=True in the definition of CIFAR10 from dataset.py

### Training CIFAR10 model with ResNet architechture
    python train.py --cuda --outf $output dir$ --dataroot dataset --archi resnet34 --batchSize 100 --gpu 0 --niter 3700

### Training class$0-9$ model of CIFAR10 with WideResnet architechture
    python train.py --cuda --outf $output dir$ --dataroot dataset --archi wideresnet --batchSize 100 --gpu 0 --niter 4500 --class_num $0-9$ --wgtDecay 0.0055

### Generating TNR and AUROC results (Table 1) for CIFAR10 and AUROC for class-wise models of CIFAR10 (Table 2)
Table 1 results, For ICAD --n 1, for ours --n 20, assumes CIFAR10 model saved as saved_models/cifar10.pth:

    python check_OOD.py --cuda --dataroot dataset --batchSize 100 --gpu 0 --net saved_models/cifar10.pth --n $1,20$ --ood_dataset $SVHN/LSUN/CIFAR100/Places365$ --archi resnet34 --one_class_det 0 --l 9000

Table 1 baseline results for CIFAR100: 

    python check_OOD.py --cuda --dataroot dataset --batchSize 100 --gpu 0 --net saved_models/cifar10.pth --n 1 --ood_dataset CFIAR100 --archi resnet34 --one_class_det 0 --l 9000

    python gen_baseline_results.py

Table 2 results, assumes that the models are saved as saved_models/class$0-9$.pth and saves the results for all the classes in all_results_valsize_900.txt

    python check_OOD.py --cuda --dataroot dataset --batchSize 100 --gpu 0  --n 20 --indist_class 10 --archi wideresnet  --l 900

### Generating TNR/AUROC VS n plots for CIFAR10 dataset, TNR results are saved in CIFAR10_$SVHN/LSUN/CIFAR100/Places365$_avr_tnr_diff_n_50.npz and AUROC results saved in CIFAR10_$SVHN/LSUN/CIFAR100/Places365$_avr_roc_diff_n_50.npz
    python check_performance_n.py --cuda --dataroot dataset --batchSize 100 --gpu 0 --net saved_models/cifar10.pth --n 50 --ood_dataset $SVHN/LSUN/CIFAR100/Places365$ --l 9000

### Link to the saved models
    https://drive.google.com/drive/folders/1AOA-xHDc5Wlh3gycC09iDW2h9NqZPe8R?usp=sharing
