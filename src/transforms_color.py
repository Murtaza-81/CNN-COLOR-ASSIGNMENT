"""Color space transformations"""

import numpy as np
import torchvision.transforms as transforms
import torch

class ToHSV:
    """Convert RGB to HSV color space"""
    def __call__(self, img):
        img = np.array(img, dtype=np.float32) / 255.0
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        diff = max_rgb - min_rgb
        
        # Compute Hue
        h = np.zeros_like(max_rgb)
        mask = diff != 0
        r_mask = (max_rgb == r) & mask
        g_mask = (max_rgb == g) & mask
        b_mask = (max_rgb == b) & mask
        
        h[r_mask] = (60 * ((g[r_mask] - b[r_mask]) / diff[r_mask])) % 360
        h[g_mask] = (60 * ((b[g_mask] - r[g_mask]) / diff[g_mask]) + 120) % 360
        h[b_mask] = (60 * ((r[b_mask] - g[b_mask]) / diff[b_mask]) + 240) % 360
        
        # Compute Saturation
        s = np.zeros_like(max_rgb)
        s[max_rgb != 0] = diff[max_rgb != 0] / max_rgb[max_rgb != 0]
        
        # Value
        v = max_rgb
        
        # Normalize to [0, 1]
        hsv = np.stack([h/360.0, s, v], axis=2)
        return transforms.ToTensor()(hsv)

class ToLAB:
    """Convert RGB to LAB color space (approximation)"""
    def __call__(self, img):
        img = np.array(img, dtype=np.float32) / 255.0
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        
        # RGB to XYZ
        r = np.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
        g = np.where(g > 0.04045, ((g + 0.055) / 1.055) ** 2.4, g / 12.92)
        b = np.where(b > 0.04045, ((b + 0.055) / 1.055) ** 2.4, b / 12.92)
        
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        # XYZ to LAB
        x, y, z = x / 0.950456, y / 1.0, z / 1.088754
        x = np.where(x > 0.008856, x ** (1/3), (7.787 * x) + 16/116)
        y = np.where(y > 0.008856, y ** (1/3), (7.787 * y) + 16/116)
        z = np.where(z > 0.008856, z ** (1/3), (7.787 * z) + 16/116)
        
        l = (116 * y) - 16
        a = 500 * (x - y)
        b = 200 * (y - z)
        
        # Normalize to [0, 1]
        lab = np.stack([l/100.0, (a+128)/256.0, (b+128)/256.0], axis=2)
        return transforms.ToTensor()(lab)

def get_train_transform(color_space):
    """Get training transform with augmentation"""
    if color_space == "rgb":
        base = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    elif color_space == "hsv":
        base = transforms.Compose([ToHSV(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif color_space == "lab":
        base = transforms.Compose([ToLAB(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        raise ValueError(f"Unknown color space: {color_space}")
    
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        base
    ])

def get_val_transform(color_space):
    """Get validation/test transform (no augmentation)"""
    if color_space == "rgb":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    elif color_space == "hsv":
        return transforms.Compose([ToHSV(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif color_space == "lab":
        return transforms.Compose([ToLAB(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])