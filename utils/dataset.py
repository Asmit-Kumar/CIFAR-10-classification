"""
CIFAR-10 Data Pipeline Utilities.

Provides reusable functions for loading and transforming the CIFAR-10 dataset
with configurable augmentation, train/val splitting, and DataLoader creation.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ── Normalization Constants ─────────────────────────────────────────────────────

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# ── Transforms ──────────────────────────────────────────────────────────────────

def get_cifar10_transforms(img_size: int = 32, use_imagenet_stats: bool = True):
    """
    Build train and validation/test transforms for CIFAR-10.

    Args:
        img_size: Target image resolution. When >32, a Resize is prepended
                  (useful for transfer learning with ImageNet-pretrained models).
        use_imagenet_stats: If True, normalizes with ImageNet stats (recommended
                           when using pretrained backbones). Otherwise uses
                           CIFAR-10 dataset stats.

    Returns:
        (train_transform, val_transform)
    """
    mean = IMAGENET_MEAN if use_imagenet_stats else CIFAR10_MEAN
    std = IMAGENET_STD if use_imagenet_stats else CIFAR10_STD

    # Crop padding scales with image size
    crop_padding = 4 if img_size <= 32 else int(img_size * 0.125)

    train_ops = []
    val_ops = []

    # Resize if target is larger than native 32x32
    if img_size > 32:
        train_ops.append(transforms.Resize(img_size))
        val_ops.append(transforms.Resize(img_size))

    train_ops.extend([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=crop_padding),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.5),
    ])

    val_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(train_ops), transforms.Compose(val_ops)


# ── DataLoaders ─────────────────────────────────────────────────────────────────

def get_cifar10_dataloaders(
    batch_size: int = 256,
    num_workers: int = 4,
    img_size: int = 32,
    use_imagenet_stats: bool = True,
    val_split: float = 0.2,
    data_root: str = './data',
):
    """
    Create CIFAR-10 train/val/test DataLoaders with a single call.

    The training set is split into train and validation subsets using a
    random split (default 80/20). The test set uses the official CIFAR-10
    test split.

    Args:
        batch_size: Batch size for all loaders.
        num_workers: Number of data-loading workers.
        img_size: Target image resolution (see get_cifar10_transforms).
        use_imagenet_stats: Normalize with ImageNet stats if True.
        val_split: Fraction of training data to use for validation.
        data_root: Root directory for the CIFAR-10 dataset.

    Returns:
        (trainloader, valloader, testloader)
    """
    train_tf, val_tf = get_cifar10_transforms(img_size, use_imagenet_stats)

    # Download once (raw, no transform)
    full_trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True,
    )

    # Random train/val split
    train_idx, val_idx = torch.utils.data.random_split(
        range(len(full_trainset)), [1 - val_split, val_split],
    )

    trainset = Subset(
        torchvision.datasets.CIFAR10(root=data_root, train=True, transform=train_tf),
        train_idx.indices,
    )
    valset = Subset(
        torchvision.datasets.CIFAR10(root=data_root, train=True, transform=val_tf),
        val_idx.indices,
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    valloader = DataLoader(
        valset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=val_tf,
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    return trainloader, valloader, testloader
