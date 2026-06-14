"""
CIFAR-10 Data Pipeline Utilities.

Provides reusable functions for loading and transforming the CIFAR-10 dataset
with configurable augmentation, train/val splitting, and DataLoader creation.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
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

IMAGENET100_CLASSES = (
    'American alligator, Alligator mississipiensis', 'American coot, marsh hen, mud hen, water hen, Fulica americana',
    'Dungeness crab, Cancer magister', 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
    'agama', 'albatross, mollymawk', 'axolotl, mud puppy, Ambystoma mexicanum', 'bald eagle, American eagle, Haliaeetus leucocephalus',
    'banded gecko', 'barn spider, Araneus cavaticus', 'bee eater', 'bittern', 'black and gold garden spider, Argiope aurantia',
    'black grouse', 'black swan, Cygnus atratus', 'black widow, Latrodectus mactans', 'boa constrictor, Constrictor constrictor',
    'bulbul', 'bustard', 'chambered nautilus, pearly nautilus, nautilus', 'chickadee', 'chiton, coat-of-mail shell, sea cradle, polyplacophore',
    'common iguana, iguana, Iguana iguana', 'common newt, Triturus vulgaris', 'conch', 'coucal', 'crane', 'crayfish, crawfish, crawdad, crawdaddy',
    'diamondback, diamondback rattlesnake, Crotalus adamanteus', 'drake', 'electric ray, crampfish, numbfish, torpedo', 'flamingo',
    'flatworm, platyhelminth', 'garden spider, Aranea diademata', 'garter snake, grass snake', 'goldfinch, Carduelis carduelis',
    'goldfish, Carassius auratus', 'goose', 'great grey owl, great gray owl, Strix nebulosa', 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
    'green lizard, Lacerta viridis', 'green mamba', 'green snake, grass snake', 'hammerhead, hammerhead shark', 'harvestman, daddy longlegs, Phalangium opilio',
    'hen', 'hermit crab', 'hognose snake, puff adder, sand viper', 'hornbill', 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
    'hummingbird', 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'jellyfish', 'king snake, kingsnake', 'kite',
    'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 'limpkin, Aramus pictus', 'loggerhead, loggerhead turtle, Caretta caretta',
    'lorikeet', 'macaw', 'magpie', 'mud turtle', 'nematode, nematode worm, roundworm', 'night snake, Hypsiglena torquata', 'oystercatcher, oyster catcher',
    'peacock', 'pelican', 'prairie chicken, prairie grouse, prairie fowl', 'ptarmigan', 'red-backed sandpiper, dunlin, Erolia alpina',
    'redshank, Tringa totanus', 'rock crab, Cancer irroratus', 'rooster', 'scorpion', 'sea anemone, anemone', 'sea lion', 'sea slug, nudibranch',
    'sea snake', 'sidewinder, horned rattlesnake, Crotalus cerastes', 'snail', 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
    'spoonbill', 'spotted salamander, Ambystoma maculatum', 'stingray', 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
    'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 'tarantula', 'tench, Tinca tinca', 'terrapin', 'thunder snake, worm snake, Carphophis amoenus',
    'tick', 'tiger shark, Galeocerdo cuvieri', 'toucan', 'vine snake', 'wallaby, brush kangaroo', 'water ouzel, dipper', 'whiptail, whiptail lizard',
    'white stork, Ciconia ciconia', 'wolf spider, hunting spider', 'wombat'
)


# ── Transforms ──────────────────────────────────────────────────────────────────

def get_cifar10_transforms(
        img_size: int = 32,
        use_imagenet_stats: bool = True,
        random_erasing_prob: float = 0.25,
        interpolation = transforms.InterpolationMode.BILINEAR,
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
        train_ops.append(transforms.Resize(img_size, interpolation=interpolation))
        val_ops.append(transforms.Resize(img_size, interpolation=interpolation))

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
        interpolation = transforms.InterpolationMode.BILINEAR,
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
        train_ops.append(transforms.Resize(img_size, interpolation=interpolation))
        val_ops.append(transforms.Resize(img_size, interpolation=interpolation))

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
    interpolation = transforms.InterpolationMode.BILINEAR,
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

    train_tf, val_tf = get_cifar10_transforms(
        img_size=img_size,
        use_imagenet_stats=use_imagenet_stats,
        interpolation=interpolation
    )

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
    interpolation = transforms.InterpolationMode.BILINEAR,
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
        interpolation: Interpolation mode for CIFAR-100 dataset upscaling.

    Returns:
        (trainloader, valloader, testloader)
    """
    if data_root is None:
        data_root = _get_default_data_root()

    train_tf, val_tf = get_cifar100_transforms(
        img_size=img_size,
        use_imagenet_stats=use_imagenet_stats,
        interpolation=interpolation
    )

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


# ── ImageNet-100 ───────────────────────────────────────────────────────────────

def get_imagenet100_transforms(
        img_size: int = 224,
        rand_augment_ops: int = 2,
        rand_augment_magnitude: int = 9,
        random_erasing_prob: float = 0.25,
        interpolation = transforms.InterpolationMode.BILINEAR,
):
    """
    Build train and validation/test transforms for ImageNet-100.

    Args:
        img_size:               Target image resolution (standard is 224).
        rand_augment_ops:       Number of sequential RandAugment operations.
        rand_augment_magnitude: Distortion severity for each operation.
        random_erasing_prob:    Probability of applying CutOut-style random erasing.

    Returns:
        (train_transform, val_transform)
    """
    train_ops = [
        transforms.RandomResizedCrop(img_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(
            num_ops=rand_augment_ops,
            magnitude=rand_augment_magnitude,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.20)),
    ]

    val_ops = [
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return transforms.Compose(train_ops), transforms.Compose(val_ops)


class HFPyTorchDataset(Dataset):
    """
    Standard PyTorch Dataset wrapper for Hugging Face datasets.
    Bypasses HF's `with_transform` to prevent memory leaks and time
    inconsistencies caused by PyArrow and DataLoader multiprocessing on Windows.
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, item["label"]


def get_imagenet100_dataloaders(
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
    data_root: str = None,
    rand_augment_ops: int = 2,
    rand_augment_magnitude: int = 9,
    random_erasing_prob: float = 0.25,
    interpolation = transforms.InterpolationMode.BILINEAR,
):
    """
    Create ImageNet-100 train/val/test DataLoaders using Hugging Face datasets.

    Loads 'ilee0022/ImageNet100' which has 'train', 'validation', and 'test' splits.
    The validation and test splits are exchanged to naturally use the 5,000-image
    split for validation and the 13,000-image split for testing.

    Args:
        batch_size:             Batch size for all loaders.
        num_workers:            Number of data-loading workers.
        img_size:               Target image resolution (standard is 224).
        data_root:              Root directory for the datasets (defaults to ./data).
        rand_augment_ops:       Number of sequential RandAugment operations.
        rand_augment_magnitude: Distortion severity for each operation.
        random_erasing_prob:    Probability of applying CutOut-style random erasing.

    Returns:
        (trainloader, valloader, testloader)
    """
    from datasets import load_dataset
    from pathlib import Path

    if data_root is None:
        data_root = _get_default_data_root()

    # Place Hugging Face cache files locally under ./data/huggingface
    hf_cache_dir = str(Path(data_root) / "huggingface")

    # 1. Download and load raw dataset from Hugging Face Hub (cached locally)
    print(f"[Dataset] Loading 'ilee0022/ImageNet100' (local cache: {hf_cache_dir})...")
    raw_dataset = load_dataset("ilee0022/ImageNet100", cache_dir=hf_cache_dir)

    train_dataset = raw_dataset["train"]
    # Exchange splits: use raw test split (5,000 images) for validation,
    # and raw validation split (13,000 images) for testing.
    val_dataset = raw_dataset["test"]
    test_dataset = raw_dataset["validation"]

    # 2. Create transforms
    train_tf, val_tf = get_imagenet100_transforms(
        img_size=img_size,
        rand_augment_ops=rand_augment_ops,
        rand_augment_magnitude=rand_augment_magnitude,
        random_erasing_prob=random_erasing_prob,
        interpolation=interpolation,
    )

    # 3. Wrap Hugging Face datasets in standard PyTorch Dataset
    train_dataset = HFPyTorchDataset(train_dataset, transform=train_tf)
    val_dataset = HFPyTorchDataset(val_dataset, transform=val_tf)
    test_dataset = HFPyTorchDataset(test_dataset, transform=val_tf)

    # 4. Create DataLoaders
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    test_workers = min(2, num_workers)
    test_loader_kwargs = {
        "num_workers": test_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if test_workers > 0:
        test_loader_kwargs["persistent_workers"] = True

    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        **loader_kwargs,
    )
    valloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        **loader_kwargs,
    )
    testloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        **test_loader_kwargs,
    )

    return trainloader, valloader, testloader
