import os
import ssl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Avoid macOS python default SSL certificate verification errors for pretrained weights
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

class CustomBrainCNN(nn.Module):
    """
    Model 1: Custom 2D-CNN Architecture Designed from Scratch.
    Consists of 4 Convolutional Blocks with Batch Normalization, Max Pooling,
    and Dropout, followed by a multi-layer Dense classifier head.
    """
    def __init__(self, num_classes=4, in_channels=3):
        super(CustomBrainCNN, self).__init__()
        self.model_name = "Custom 2D-CNN (Designed Architecture)"
        
        # Block 1: 3 -> 32 -> 32
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112
            nn.Dropout2d(0.1)
        )
        
        # Block 2: 32 -> 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56
            nn.Dropout2d(0.15)
        )
        
        # Block 3: 64 -> 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28
            nn.Dropout2d(0.2)
        )
        
        # Block 4: 128 -> 256
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
            nn.Dropout2d(0.25)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))  # 256 * 2 * 2 = 1024
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class PretrainedResNet18(nn.Module):
    """
    Model 2: Pretrained 2D-CNN (ResNet18 Transfer Learning).
    Leverages ImageNet pre-trained representations and fine-tunes with a custom
    deep classification head matching the repository research design.
    """
    def __init__(self, num_classes=4, pretrained=True, freeze_backbone=False):
        super(PretrainedResNet18, self).__init__()
        self.model_name = "Pretrained ResNet18 (Transfer Learning)"
        
        # Initialize ResNet18 backbone
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            self.backbone = models.resnet18(pretrained=pretrained)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        num_ftrs = self.backbone.fc.in_features
        
        # Custom classification head as proposed in the repo
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class PretrainedResNet50(nn.Module):
    """
    Alternative Pretrained 2D-CNN: ResNet50 Transfer Learning.
    """
    def __init__(self, num_classes=4, pretrained=True, freeze_backbone=False):
        super(PretrainedResNet50, self).__init__()
        self.model_name = "Pretrained ResNet50 (Transfer Learning)"
        
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        except Exception:
            self.backbone = models.resnet50(pretrained=pretrained)
            
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        num_ftrs = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def get_model(model_type="custom_cnn", num_classes=4, pretrained=True):
    """
    Factory method to instantiate models.
    """
    if model_type == "custom_cnn":
        return CustomBrainCNN(num_classes=num_classes)
    elif model_type == "resnet18":
        return PretrainedResNet18(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "resnet50":
        return PretrainedResNet50(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
