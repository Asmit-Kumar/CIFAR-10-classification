"""
CIFAR-10 / CIFAR-100 Shared Utilities.

Submodules:
    dataset   — Data loading, transforms, normalization constants (CIFAR-10 & CIFAR-100)
    training  — Training loops, evaluation, per-class accuracy
    plotting  — Training curves, image visualization
    callbacks — EarlyStopping, ModelCheckpoint
    logger    — RunLogger: lightweight experiment tracker (JSON-backed)
"""

from .dataset import (
    # Normalization constants
    CIFAR10_MEAN, CIFAR10_STD,
    CIFAR100_MEAN, CIFAR100_STD,
    IMAGENET_MEAN, IMAGENET_STD,
    # Class label tuples
    CIFAR10_CLASSES, CIFAR100_CLASSES,
    CLASSES,  # alias for CIFAR10_CLASSES (backward-compat)
    # DataLoader factories
    get_cifar10_transforms, get_cifar10_dataloaders,
    get_cifar100_transforms, get_cifar100_dataloaders,
    get_imagenet100_transforms, get_imagenet100_dataloaders,
)
from .training import train_one_epoch, evaluate, test_accuracy, fit
from .plotting import plot_training_curves, imshow, show_sample_batch
from .callbacks import EarlyStopping, ModelCheckpoint
from .logger import RunLogger
