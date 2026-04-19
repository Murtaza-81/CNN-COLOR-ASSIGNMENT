"""Utility functions and configuration"""

import os
import random
import json
import numpy as np
import torch
from datetime import datetime

def set_seed(seed=42):
    """Fix random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    """Config class for all experiments"""
    # Training parameters
    MAX_EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 10
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 3
    MIN_DELTA = 0.001
    
    # Data
    DATA_DIR = "./data"
    VAL_SPLIT = 0.1
    
    # Output directories
    OUTPUT_DIR = "outputs"
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CIFAR-10 classes
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    @classmethod
    def create_dirs(cls):
        """Create output directories"""
        for dir_path in [cls.PLOTS_DIR, cls.LOGS_DIR, cls.CHECKPOINTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print config settings"""
        print("="*70)
        print("Configuration Settings")
        print("="*70)
        print(f"Device: {cls.DEVICE}")
        print(f"Max Epochs: {cls.MAX_EPOCHS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"Early Stopping Patience: {cls.EARLY_STOPPING_PATIENCE}")
        print("="*70)

def save_metrics(metrics, filename):
    """Save metrics to JSON file"""
    with open(os.path.join(Config.LOGS_DIR, filename), "w") as f:
        json.dump(metrics, f, indent=2)

def load_metrics(filename):
    """Load metrics from JSON file"""
    with open(os.path.join(Config.LOGS_DIR, filename), "r") as f:
        return json.load(f)