import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models import resnet18, resnet50

from typing import *

class ResNet18(nn.Module): 
    '''
    Resnet18 for CIFAR. Conv1 and maxpooling is reset because Pytorch's implementation
    is used for ImageNet training which will hurt performance on 32 * 32 images.
    Args:
        num_classes : [number of fine classes]; or in multi task learning, to
        use two fully connected heads, specify the size of layer by [number of
        coarse classes, number of fine classes]
    '''
    def __init__(self, num_classes : List[int]):
        super(ResNet18, self).__init__()
        # fine-tune
        self.backbone = resnet18(weights='IMAGENET1K_V1')
        modules = list(self.backbone.children())[:-1] # remove last linear layer
        for name, p in self.backbone.named_parameters():
            if not name.startswith('layer4'):
                p.requires_grad = False
        self.base = nn.Sequential(*modules)
        self.out_features = 512 # hard-coded
        self.head = len(num_classes)
        if self.head == 1:
            self.fc = nn.Linear(self.out_features, num_classes[0]) 
        elif self.head == 2:
            num_coarse_classes = num_classes[0]
            num_fine_classes = num_classes[1]
            assert (num_coarse_classes <= num_fine_classes), 'invalid hierarchy'
            self.fc1 = nn.Linear(self.out_features, num_coarse_classes) 
            self.fc2 = nn.Linear(self.out_features, num_fine_classes)
        
    def forward(self, x : Tensor) -> Tuple[Tensor, ...]:
        # representation vector, returned here to avoid extra gpu usage
        x = torch.squeeze(self.base(x)) 
        if self.head == 1:
            return x, self.fc(x)
        elif self.head == 2:
            return x, self.fc1(x), self.fc2(x)

class ResNet50(nn.Module): 
    def __init__(self, num_classes : List[int]):
        super(ResNet50, self).__init__()
        self.backbone = resnet50(weights='IMAGENET1K_V1')
        modules = list(self.backbone.children())[:-1] # remove last linear layer
        for name, p in self.backbone.named_parameters():
            if not name.startswith('layer4'):
                p.requires_grad = False
        self.base = nn.Sequential(*modules)
        self.out_features = 2048 # hard-coded
        self.head = len(num_classes)
        if self.head == 1:
            self.fc = nn.Linear(self.out_features, num_classes[0]) 
        elif self.head == 2:
            num_coarse_classes = num_classes[0]
            num_fine_classes = num_classes[1]
            assert (num_coarse_classes <= num_fine_classes), 'invalid hierarchy'
            self.fc1 = nn.Linear(self.out_features, num_coarse_classes) 
            self.fc2 = nn.Linear(self.out_features, num_fine_classes)
        
    def forward(self, x : Tensor) -> Tuple[Tensor, ...]:
        # representation vector, returned here to avoid extra gpu usage
        x = torch.squeeze(self.base(x)) 
        if self.head == 1:
            return x, self.fc(x)
        elif self.head == 2:
            return x, self.fc1(x), self.fc2(x)