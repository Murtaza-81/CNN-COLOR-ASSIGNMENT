"""Evaluation module for trained models"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from .utils import Config, set_seed
from .data import load_cifar10
from models import CNNBaseline, CNNExtension

def load_model(checkpoint_path, use_extension=False):
    """Load trained model from checkpoint"""
    if use_extension:
        model = CNNExtension(num_classes=Config.NUM_CLASSES)
    else:
        model = CNNBaseline(num_classes=Config.NUM_CLASSES)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model = model.to(Config.DEVICE)
    model.eval()
    return model

def analyze_channel_mixing(model, color_space="rgb"):
    """Analyze and visualize channel mixing in first convolutional layer"""
    print(f"\n{'='*60}")
    print("Channel Mixing Analysis")
    print(f"{'='*60}")
    
    # Extract first conv layer weights
    if hasattr(model, 'model'):
        first_conv = model.model.conv1
    elif hasattr(model, 'cnn'):
        first_conv = model.cnn.conv1
    else:
        first_conv = model.model.conv1
    
    weights = first_conv.weight.data.cpu().numpy()
    out_channels, in_channels = weights.shape[0], weights.shape[1]
    
    # Compute per-filter channel energy
    channel_energy = np.sum(weights ** 2, axis=(2, 3))
    channel_energy_norm = channel_energy / (channel_energy.sum(axis=1, keepdims=True) + 1e-8)
    
    # Plot channel energy heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(channel_energy, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Input Channel')
    axes[0].set_ylabel('Filter Index')
    axes[0].set_title(f'Channel Energy Heatmap - {color_space.upper()}')
    axes[0].set_xticks(range(in_channels))
    axes[0].set_xticklabels([f'Ch{i+1}' for i in range(in_channels)])
    plt.colorbar(im1, ax=axes[0], label='Energy')
    
    im2 = axes[1].imshow(channel_energy_norm, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xlabel('Input Channel')
    axes[1].set_ylabel('Filter Index')
    axes[1].set_title(f'Normalized Channel Energy - {color_space.upper()}')
    axes[1].set_xticks(range(in_channels))
    axes[1].set_xticklabels([f'Ch{i+1}' for i in range(in_channels)])
    plt.colorbar(im2, ax=axes[1], label='Normalized Energy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, f"channel_energy_heatmap_{color_space}.png"), 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nFilter channel bias analysis:")
    for ch in range(in_channels):
        top_filters = np.argsort(channel_energy_norm[:, ch])[-5:][::-1]
        print(f"Top 5 filters biased to Channel {ch+1}: {top_filters}")
    
    return channel_energy_norm

def plot_confusion_matrix(model, test_loader, color_space):
    """Plot confusion matrix for the model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(Config.DEVICE)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=Config.CLASSES, yticklabels=Config.CLASSES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {color_space.upper()}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, f"confusion_matrix_{color_space}.png"), 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nClassification Report - {color_space.upper()}:")
    print(classification_report(all_labels, all_preds, target_names=Config.CLASSES))
    
    return cm

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CNN model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--colorspace", type=str, default="rgb",
                        help="Color space used for training")
    parser.add_argument("--analyze-channels", action="store_true",
                        help="Perform channel mixing analysis")
    parser.add_argument("--plot-confusion", action="store_true",
                        help="Plot confusion matrix")
    
    args = parser.parse_args()
    
    set_seed(42)
    Config.create_dirs()
    
    # Load model
    use_extension = "extension" in args.checkpoint
    model = load_model(args.checkpoint, use_extension)
    
    # Load data
    _, _, test_loader = load_cifar10(args.colorspace)
    
    # Evaluate
    if args.analyze_channels:
        analyze_channel_mixing(model, args.colorspace)
    
    if args.plot_confusion:
        plot_confusion_matrix(model, test_loader, args.colorspace)

if __name__ == "__main__":
    main()