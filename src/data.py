"""Data loading and preparation"""

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from .transforms_color import get_train_transform, get_val_transform
from .utils import Config

def load_cifar10(color_space="rgb"):
    """Load CIFAR-10 dataset with specified color space"""
    
    train_transform = get_train_transform(color_space)
    val_transform = get_val_transform(color_space)
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=Config.DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=Config.DATA_DIR, train=False, download=True, transform=val_transform
    )
    
    # Split training into train/val
    val_size = int(len(train_dataset) * Config.VAL_SPLIT)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def get_dataset_info():
    """Get dataset information"""
    return {
        "name": "CIFAR-10",
        "num_classes": 10,
        "image_size": (32, 32),
        "classes": Config.CLASSES
    }