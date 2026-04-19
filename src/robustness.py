"""Robustness benchmarking module"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import Config, set_seed, save_metrics
from .data import load_cifar10
from .eval import load_model

def gamma_correction(images, gamma):
    """Apply gamma correction"""
    return images ** gamma

def brightness_contrast(images, alpha=1.0, beta=0.0):
    """Adjust brightness and contrast"""
    return torch.clamp(alpha * images + beta, 0, 1)

def channel_dropout(images, channel_idx):
    """Drop a specific channel"""
    images_dropped = images.clone()
    images_dropped[:, channel_idx] = 0
    return images_dropped

def color_temperature(images, temperature_factor):
    """Adjust color temperature"""
    images_adjusted = images.clone()
    if temperature_factor > 1:
        images_adjusted[:, 0] = torch.clamp(images[:, 0] * temperature_factor, 0, 1)
        images_adjusted[:, 2] = torch.clamp(images[:, 2] / temperature_factor, 0, 1)
    else:
        images_adjusted[:, 2] = torch.clamp(images[:, 2] / temperature_factor, 0, 1)
        images_adjusted[:, 0] = torch.clamp(images[:, 0] * temperature_factor, 0, 1)
    return images_adjusted

def evaluate_robustness(model, test_loader, severity_levels=None):
    """Evaluate model robustness under various perturbations"""
    
    if severity_levels is None:
        severity_levels = [0, 1, 2, 3, 4, 5]
    
    perturbations = {
        'Gamma Correction': {
            'func': gamma_correction,
            'levels': [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
        },
        'Brightness/Contrast': {
            'func': brightness_contrast,
            'levels': severity_levels
        },
        'Channel Dropout': {
            'func': channel_dropout,
            'levels': severity_levels[:5]
        },
        'Color Temperature': {
            'func': color_temperature,
            'levels': [0.5, 0.7, 1.0, 1.5, 2.0, 2.5]
        }
    }
    
    model.eval()
    results = {}
    
    for pert_name, pert_config in perturbations.items():
        print(f"\nTesting: {pert_name}")
        severity_levels = pert_config['levels']
        accuracies = []
        
        for severity in severity_levels:
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader, desc=f"Severity {severity}", leave=False):
                    inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                    
                    if pert_name == 'Gamma Correction':
                        perturbed = pert_config['func'](inputs, severity)
                    elif pert_name == 'Brightness/Contrast':
                        alpha = 0.5 + severity * 0.3
                        beta = severity * 0.1
                        perturbed = pert_config['func'](inputs, alpha, beta)
                    elif pert_name == 'Channel Dropout':
                        channel = int(severity % 3)
                        perturbed = pert_config['func'](inputs, channel)
                    elif pert_name == 'Color Temperature':
                        temp_factor = 0.5 + severity * 0.4
                        perturbed = pert_config['func'](inputs, temp_factor)
                    else:
                        perturbed = inputs
                    
                    outputs = model(perturbed)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            accuracies.append(acc)
            print(f"  Severity {severity}: {acc:.2f}%")
        
        results[pert_name] = {
            'severity_levels': severity_levels,
            'accuracies': accuracies
        }
    
    return results

def plot_robustness_curves(results):
    """Plot robustness curves for all perturbations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for idx, (pert_name, data) in enumerate(results.items()):
        ax = axes[idx]
        ax.plot(data['severity_levels'], data['accuracies'], 
                marker='o', linewidth=2, markersize=8, color=colors[idx])
        ax.set_xlabel('Severity Level')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Robustness to {pert_name}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        for i, (sev, acc) in enumerate(zip(data['severity_levels'], data['accuracies'])):
            ax.annotate(f'{acc:.1f}%', (sev, acc), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=9)
    
    plt.suptitle('Model Robustness to Illumination and Color Shifts', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, "robustness_curves.png"), dpi=150, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate model robustness")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--colorspace", type=str, default="rgb",
                        help="Color space used for training")
    parser.add_argument("--severity-levels", type=int, nargs='+', 
                        default=[0, 1, 2, 3, 4, 5],
                        help="Severity levels to test")
    
    args = parser.parse_args()
    
    set_seed(42)
    Config.create_dirs()
    
    # Load model
    use_extension = "extension" in args.checkpoint
    model = load_model(args.checkpoint, use_extension)
    
    # Load data
    _, _, test_loader = load_cifar10(args.colorspace)
    
    # Evaluate robustness
    results = evaluate_robustness(model, test_loader, args.severity_levels)
    
    # Plot results
    plot_robustness_curves(results)
    
    # Save results
    save_metrics(results, "robustness_metrics.json")
    
    print("\n✓ Robustness evaluation completed!")

if __name__ == "__main__":
    main()