from typing import *

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from hierarchy.data import Hierarchy, get_k_shot

class HierarchyCIFAR10(Dataset):
    def __init__(self, train, split):
        self.dataset_name = 'CIFAR10'
        if split == 'full':
            self.fine_names = ['airplane','automobile','bird','cat','deer',
                        'dog','frog','horse','ship','truck']
            self.coarse_names = ['animal', 'transport']
            self.fine_map = np.arange(10)
            self.coarse_map = np.array([1,1,0,0,0,0,0,0,1,1])
        elif split == 'source':
            self.fine_names = ['deer','dog','frog','horse','ship','truck']
            self.coarse_names = ['animal', 'transport']
            self.fine_map = np.arange(6)
            self.coarse_map = np.array([0,0,0,0,1,1])
        elif split == 'target':
            self.fine_names = ['airplane','automobile','bird','cat']
            self.coarse_names = ['animal', 'transport']
            self.fine_map = np.arange(4)
            self.coarse_map = np.array([1,1,0,0])

        self.split = split
        self.img_size = 32
        self.channel = 3
        self.train = train
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
                transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        cifar10_base = CIFAR10(root = './data',train = self.train, download=True) 
        cifar10_base.targets = np.array(cifar10_base.targets)                   
        fine_idx = [cifar10_base.class_to_idx[n] for n in self.fine_names]
        reset_idx_map = {idx:i for i,idx in enumerate(fine_idx)}

        target_idx = np.concatenate([np.argwhere(cifar10_base.targets == i).flatten() for i in fine_idx])
        self.data = cifar10_base.data[target_idx]
        self.targets = cifar10_base.targets[target_idx]
        self.targets = [reset_idx_map[i] for i in self.targets]

        self.mid_map = None
        self.coarsest_map = None
        self.mid2coarse = None
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index: int):
        img, target_fine = self.data[index], int(self.targets[index])

        target_mid = -1
        target_coarse = int(self.coarse_map[target_fine])
        target_coarsest = -1
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target_coarsest, target_coarse, target_mid, target_fine


def make_dataloader(num_workers : int, batch_size : int, task : str) -> Tuple[DataLoader, DataLoader]:
    '''
    Creat (a subset of) train test dataloader. Train & test has the same number of classes.
    Args:
        num_workers : number of workers of train and test loader.
        batch_size : batch size of train and test loader
        task : if 'split_pretrain', dataset has 60 classes, if 'split_downstream',
        dataset has 40 classes, if 'full', dataset has 100 classes.
    '''

    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def make_full_dataset() -> Tuple[Dataset, Dataset]:
        '''
        Create full size augmented CIFAR100 dataset that contains four layers in
        hierarchy labels. Return a tuple of train, test dataset.
        '''
        train_dataset = HierarchyCIFAR10(train = True, split = 'full')
        test_dataset = HierarchyCIFAR10(train = False, split = 'full')
        return train_dataset, test_dataset

    def make_subpopsplit_dataset(train_dataset : Hierarchy, test_dataset : Hierarchy, task : str) -> Tuple[DataLoader, DataLoader]: 
        '''
        Given full size CIFAR100 datasets, split fine classes into 40/60. Specifically,
        For each coarse class, its 5 fine children is divided into 2 (target) and 3 (source)
        in an increasing alphabetical order. Finally, we have 60 fine classes in source,
        40 fine classes in target. The same procedure is applied on full size train and 
        test set. 
        Depending on the task, we use different source/target train/test combination. For
        pretrain, we use 'ss'; for evaluation of representation on one-shot transfer, we 
        use 'tt'.
        Args:
            train_dataset : augmented CIFAR100 train dataset with four label levels
            test_dataset : normalized CIFAR100 test dataset with four label levels
            task : 'ss' = 'source train source val', 'st' = 'source train target val',
            'ts' = 'target train source val', 'tt' = 'target train target val'
        '''
        
        source_train = HierarchyCIFAR10(train=True, split="source")
        source_test = HierarchyCIFAR10(train=False, split="source")
        target_train = HierarchyCIFAR10(train=True, split="target")
        target_test = HierarchyCIFAR10(train=False, split="target")
        if task == 'ss':
            train_dataset, test_dataset = source_train, source_test
        elif task == 'st':
            train_dataset, test_dataset = source_train, target_test
        elif task == 'ts':
            train_dataset, test_dataset = target_train, source_test
        elif task == 'tt':
            train_dataset, test_dataset = target_train, target_test
        return train_dataset, test_dataset

    init_train_dataset, init_test_dataset = make_full_dataset()
    train_dataset, test_dataset = make_subpopsplit_dataset(init_train_dataset, init_test_dataset, task)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
