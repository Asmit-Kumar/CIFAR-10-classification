"""
CIFAR-10 Shared Utilities.

Submodules:
    dataset   — Data loading, transforms, normalization constants
    training  — Training loops, evaluation, per-class accuracy
    plotting  — Training curves, image visualization
    callbacks — EarlyStopping, ModelCheckpoint
"""

from .dataset import (
    CIFAR10_MEAN, CIFAR10_STD, IMAGENET_MEAN, IMAGENET_STD, CLASSES,
    get_cifar10_transforms, get_cifar10_dataloaders,
)
from .training import train_one_epoch, evaluate, test_accuracy, fit
from .plotting import plot_training_curves, imshow, show_sample_batch
from .callbacks import EarlyStopping, ModelCheckpoint
