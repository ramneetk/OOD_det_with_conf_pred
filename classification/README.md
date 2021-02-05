### Downloading CIFAR10 dataset in dataset folder
   mkdir dataset
   
   Set download=True in the definition of CIFAR10 from dataset.py

### Training AVT model with Wideresnet architechture on class$class_num$ of CIFAR10
      python main.py --cuda --outf $output dir for saving the model$ --dataroot dataset --gpu $gpu_num$ --class_num $0-9$ 

### OOD detection
Command for running experiments for all 10 classes, assumes models saved in saved_models/class$class_num$_net.pth

      python check_OOD_CEL.py --cuda --dataroot dataset --batchSize $batch_size$ --gpu $gpu_num$ --n $martingale_param n$ --indist_class 10 --l 900

Command for running experiment for a single class
      
      python check_OOD_CEL.py --cuda --dataroot dataset --batchSize $batch_size$ --gpu $gpu_num$ --n $martingale_param n$ --net $path to the saved model$ --ood_dataset cifar_non$class_num$_class  --indist_class $class_num$ --l 900 --lmbda 0

### Link to the saved models

    https://drive.google.com/drive/folders/1pMs8Mckjv3V5NjfXUhA-5ufPF8Hf0mW6?usp=sharing
