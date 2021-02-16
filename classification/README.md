### Downloading CIFAR10 dataset in dataset folder   
   Set download=True in the definition of CIFAR10 from dataset.py

### Training AVT model with Wideresnet architechture on class$class_num$ of CIFAR10
    python train.py --cuda --outf $output dir for saving the model$ --dataroot dataset --gpu $gpu_num$ --class_num $0-9$ 

### OOD detection
Command for generating results in table 3, assumes models saved in saved_models/class$class_num$.pth
results are saved in all_results_valsize_900.txt

    python check_OOD.py --cuda --dataroot dataset --batchSize $batch_size$ --gpu $gpu_num$ --n $no. of transformations$ --indist_class 10 --l 900

### Link to the saved models

    https://drive.google.com/drive/folders/1pMs8Mckjv3V5NjfXUhA-5ufPF8Hf0mW6?usp=sharing
