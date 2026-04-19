"""Baseline CNN model (ResNet-18)"""

import torch
import torch.nn as nn
import torchvision

class CNNBaseline(nn.Module):
    """ResNet-18 based CNN for CIFAR-10 classification"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        
        # Adapt for CIFAR-10 (32x32 images)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool for CIFAR-10
        
        # Add dropout for regularization
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def get_features(self, x):
        """Get features before the final FC layer"""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x