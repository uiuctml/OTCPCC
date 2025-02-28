import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, vit_b_16, vit_l_16
import torchvision.transforms as transforms

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
        self.backbone = resnet18()  
        self.backbone.conv1 =  nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        self.backbone.maxpool = nn.Identity()
        self.base = nn.Sequential(*list(self.backbone.children())[:-1])
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
        
class ResNet34(nn.Module): 
    def __init__(self, num_classes : List[int]):
        super(ResNet34, self).__init__()
        self.backbone = resnet34()  
        self.backbone.conv1 =  nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        self.backbone.maxpool = nn.Identity()
        self.base = nn.Sequential(*list(self.backbone.children())[:-1])
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
        self.backbone = resnet50()  
        self.backbone.conv1 =  nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        self.backbone.maxpool = nn.Identity()
        self.base = nn.Sequential(*list(self.backbone.children())[:-1])
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
        
class ViTB16(nn.Module): 
    def __init__(self, num_classes: List[int]):
        super(ViTB16, self).__init__()
        
        # Load the pre-trained ViT model
        self.backbone = vit_b_16(weights='IMAGENET1K_V1')
        
        # Remove the ViT's default classification head
        self.backbone.heads = nn.Identity()  # Replace the classification head with Identity
        
        # Create a new classification head for your task
        self.fc = nn.Linear(self.backbone.hidden_dim, num_classes[0])  # Output size defined by num_classes

        self.resize_transform = transforms.Resize((224, 224))
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward method to extract features and compute classification logits.
        
        Args:
            x (Tensor): Input images of shape [batch_size, 3, 224, 224]
        
        Returns:
            Tuple[Tensor, Tensor]: Class token (features) and classification logits
        """

        x = torch.stack([self.resize_transform(img) for img in x])

        # Step 1: Patch Embedding (automatically handled by model._process_input)
        x = self.backbone._process_input(x)  # [batch_size, num_patches, hidden_dim]
        
        # Step 2: Add class token
        n = x.shape[0]  # Batch size
        class_token = self.backbone.class_token.expand(n, -1, -1)  # [batch_size, 1, hidden_dim]
        x = torch.cat([class_token, x], dim=1)  # [batch_size, num_patches + 1, hidden_dim]
        
        # Step 3: Forward through the encoder (positional embedding is added here)
        x = self.backbone.encoder(x)  # [batch_size, num_patches + 1, hidden_dim]
        
        # Step 4: Extract class token (first token)
        class_token_output = x[:, 0]  # [batch_size, hidden_dim]
        
        # Step 5: Pass the class token through the custom classification head
        output_fine = self.fc(class_token_output)  # [batch_size, num_classes[0]]
        
        return class_token_output, output_fine  # Return class token and logits
    

class ViTL16(nn.Module): 
    def __init__(self, num_classes: List[int]):
        super(ViTL16, self).__init__()
        
        # Load the pre-trained ViT model
        self.backbone = vit_l_16(weights='IMAGENET1K_V1')
        
        # Remove the ViT's default classification head
        self.backbone.heads = nn.Identity()  # Replace the classification head with Identity
        
        # Create a new classification head for your task
        self.fc = nn.Linear(self.backbone.hidden_dim, num_classes[0])  # Output size defined by num_classes

        self.resize_transform = transforms.Resize((224, 224))
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward method to extract features and compute classification logits.
        
        Args:
            x (Tensor): Input images of shape [batch_size, 3, 224, 224]
        
        Returns:
            Tuple[Tensor, Tensor]: Class token (features) and classification logits
        """

        x = torch.stack([self.resize_transform(img) for img in x])

        # Step 1: Patch Embedding (automatically handled by model._process_input)
        x = self.backbone._process_input(x)  # [batch_size, num_patches, hidden_dim]
        
        # Step 2: Add class token
        n = x.shape[0]  # Batch size
        class_token = self.backbone.class_token.expand(n, -1, -1)  # [batch_size, 1, hidden_dim]
        x = torch.cat([class_token, x], dim=1)  # [batch_size, num_patches + 1, hidden_dim]
        
        # Step 3: Forward through the encoder (positional embedding is added here)
        x = self.backbone.encoder(x)  # [batch_size, num_patches + 1, hidden_dim]
        
        # Step 4: Extract class token (first token)
        class_token_output = x[:, 0]  # [batch_size, hidden_dim]
        
        # Step 5: Pass the class token through the custom classification head
        output_fine = self.fc(class_token_output)  # [batch_size, num_classes[0]]
        
        return class_token_output, output_fine  # Return class token and logits
    