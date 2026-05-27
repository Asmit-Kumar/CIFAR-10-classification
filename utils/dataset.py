"""
CIFAR-10 Data Pipeline Utilities.

Provides reusable functions for loading and transforming the CIFAR-10 dataset
with configurable augmentation, train/val splitting, and DataLoader creation.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path

def _get_default_data_root() -> str:
    """Dynamically resolve the root 'data' directory relative to this file."""
    return str(Path(__file__).resolve().parent.parent / "data")

# ── Normalization Constants ─────────────────────────────────────────────────────

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

# Keep old CLASSES for backward compatibility
CLASSES = CIFAR10_CLASSES

CIFAR100_CLASSES = (
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
)


# ── Transforms ──────────────────────────────────────────────────────────────────

def get_cifar10_transforms(
        img_size: int = 32,
        use_imagenet_stats: bool = True,
        random_erasing_prob: float = 0.25,
):
    """
    Build train and validation/test transforms for CIFAR-10.

    Args:
        img_size: Target image resolution. When >32, a Resize is prepended
                  (useful for transfer learning with ImageNet-pretrained models).
        use_imagenet_stats: If True, normalizes with ImageNet stats (recommended
                           when using pretrained backbones). Otherwise uses
                           CIFAR-10 dataset stats.
        random_erasing_prob: Probability of applying CutOut-style random
                             patch erasing. Set to 0.0 to disable.

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
        transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.20)),
    ])

    val_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(train_ops), transforms.Compose(val_ops)


def get_cifar100_transforms(
        img_size: int = 32,
        use_imagenet_stats: bool = True,
        rand_augment_ops: int = 2,
        rand_augment_magnitude: int = 9,
        random_erasing_prob: float = 0.25,
):
    """
    Build train and validation/test transforms for CIFAR-100.

    Uses RandAugment (policy-free, generalizes across all 100 classes) and
    RandomErasing (CutOut equivalent) as the primary augmentation strategy.
    AutoAugment is intentionally avoided here — its CIFAR-10 policy is
    optimised for 10 coarse classes and degrades on CIFAR-100's fine-grained
    class structure.

    Args:
        img_size:               Target image resolution. When >32, a Resize is
                                prepended (for transfer learning backbones).
        use_imagenet_stats:     Normalize with ImageNet stats if True (recommended
                                when using pretrained backbones).
        rand_augment_ops:       Number of sequential RandAugment operations.
                                Range 1–3; default 2 is the standard CIFAR setting.
        rand_augment_magnitude: Distortion severity for each operation.
                                Range 1–30; 9 is a balanced default.
        random_erasing_prob:    Probability of applying CutOut-style random
                                patch erasing. Set to 0.0 to disable.

    Returns:
        (train_transform, val_transform)
    """
    mean = IMAGENET_MEAN if use_imagenet_stats else CIFAR100_MEAN
    std = IMAGENET_STD if use_imagenet_stats else CIFAR100_STD

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
        # RandAugment: policy-free, no CIFAR-10 bias, works across all 100 classes
        transforms.RandAugment(
            num_ops=rand_augment_ops,
            magnitude=rand_augment_magnitude,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # RandomErasing: CutOut-style patch occlusion applied after ToTensor
        transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.20)),
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
    data_root: str = None,
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
    if data_root is None:
        data_root = _get_default_data_root()

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


def get_cifar100_dataloaders(
    batch_size: int = 256,
    num_workers: int = 4,
    img_size: int = 32,
    use_imagenet_stats: bool = True,
    val_split: float = 0.2,
    data_root: str = None,
):
    """
    Create CIFAR-100 train/val/test DataLoaders with a single call.

    The training set is split into train and validation subsets using a
    random split (default 80/20). The test set uses the official CIFAR-100
    test split.

    Args:
        batch_size: Batch size for all loaders.
        num_workers: Number of data-loading workers.
        img_size: Target image resolution (see get_cifar100_transforms).
        use_imagenet_stats: Normalize with ImageNet stats if True.
        val_split: Fraction of training data to use for validation.
        data_root: Root directory for the CIFAR-100 dataset.

    Returns:
        (trainloader, valloader, testloader)
    """
    if data_root is None:
        data_root = _get_default_data_root()

    train_tf, val_tf = get_cifar100_transforms(img_size, use_imagenet_stats)

    # Download once (raw, no transform)
    full_trainset = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True,
    )

    # Random train/val split
    train_idx, val_idx = torch.utils.data.random_split(
        range(len(full_trainset)), [1 - val_split, val_split],
    )

    trainset = Subset(
        torchvision.datasets.CIFAR100(root=data_root, train=True, transform=train_tf),
        train_idx.indices,
    )
    valset = Subset(
        torchvision.datasets.CIFAR100(root=data_root, train=True, transform=val_tf),
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

    testset = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=val_tf,
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    return trainloader, valloader, testloader
