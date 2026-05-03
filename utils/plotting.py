"""
Plotting Utilities for CIFAR-10 Experiments.

Provides reusable functions for visualizing training curves and sample images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision


def plot_training_curves(train_losses, val_losses, val_accuracies):
    """
    Plot training/validation loss and validation accuracy side by side.

    Args:
        train_losses: List of per-epoch training losses.
        val_losses: List of per-epoch validation losses.
        val_accuracies: List of per-epoch validation accuracies (%).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_accuracies, label='Val Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def imshow(img, mean=None, std=None):
    """
    De-normalize and display a tensor image (or grid of images).

    Args:
        img: Tensor image (C, H, W) — typically from torchvision.utils.make_grid.
        mean: Normalization mean used during preprocessing.
        std: Normalization std used during preprocessing.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = img * std_t + mean_t
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)).clip(0, 1))
    plt.axis('off')
    plt.show()


def show_sample_batch(loader, classes, mean=None, std=None, n=8):
    """
    Display a grid of sample images from a DataLoader with their labels.

    Args:
        loader: DataLoader to sample from.
        classes: Tuple of class name strings.
        mean: Normalization mean for de-normalization.
        std: Normalization std for de-normalization.
        n: Number of images to display.
    """
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images = images[:n]
    labels = labels[:n]

    imshow(torchvision.utils.make_grid(images), mean, std)
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))
