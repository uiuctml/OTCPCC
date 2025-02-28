import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import dataset.cifar100.model as cifar100
import dataset.cifar10.model as cifar10
import dataset.breeds2.model as breeds
import dataset.inat.model as inat

from typing import *
import os

def init_model(dataset : str, num_classes : List[int], device : torch.device, arch='res18'):
    '''
    Load the correct model for each dataset.
    '''
    if dataset == 'CIFAR100':
        if arch == 'res18':
            model = cifar100.ResNet18(num_classes).to(device)
        elif arch == 'res34':
            model = cifar100.ResNet34(num_classes).to(device)
        elif arch == 'res50':
            model = cifar100.ResNet50(num_classes).to(device)
        elif arch == 'vitb16':
            model = cifar100.ViTB16(num_classes).to(device)
        elif arch == 'vitl16':
            model = cifar100.ViTL16(num_classes).to(device)
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 0:
            model = DDP(model)
    elif dataset == 'CIFAR10':
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 0:
            model = DDP(cifar10.ResNet18(num_classes).to(device))
        else:
            model = cifar10.ResNet18(num_classes).to(device)
    elif dataset == 'BREEDS':
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 0:
            model = DDP(breeds.ResNet18(num_classes).to(device), find_unused_parameters=True)
        else:
            model = breeds.ResNet18(num_classes).to(device)
    elif dataset == 'INAT':
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 0:
            model = DDP(inat.ResNet18(num_classes).to(device), gradient_as_bucket_view=True)
        else:
            model = inat.ResNet18(num_classes).to(device)

    model = torch.compile(model)
    return model