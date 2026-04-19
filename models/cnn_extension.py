"""CNN with learnable color transform layer (Extension 1)"""

import torch
import torch.nn as nn
import torchvision

class CNNExtension(nn.Module):
    """CNN with learnable 1x1 color transform layer"""
    
    def __init__(self, num_classes=10, num_output_channels=3):
        super().__init__()
        
        # Learnable 1x1 color transform layer
        self.color_transform = nn.Conv2d(3, num_output_channels, kernel_size=1, bias=True)
        
        # Initialize to identity matrix
        with torch.no_grad():
            self.color_transform.weight.data = torch.eye(3, num_output_channels).view(
                num_output_channels, 3, 1, 1
            )
            if num_output_channels == 3:
                self.color_transform.bias.data.zero_()
        
        # Main CNN
        self.cnn = torchvision.models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(num_output_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn.maxpool = nn.Identity()
        self.cnn.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.color_transform(x)
        return self.cnn(x)
    
    def get_color_transform_matrix(self):
        """Get the learned color transform matrix"""
        return self.color_transform.weight.data.cpu().numpy().reshape(3, -1)