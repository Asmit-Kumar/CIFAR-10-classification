# Image Classification with PyTorch

A progressive study in image classification — building from first principles to competitive deep learning, evaluated across three datasets of increasing complexity and scale.

| Dataset | Classes | Resolution | Training Set | Best Test Acc |
|---------|---------|-----------|-------------|---------------|
| [**CIFAR-10**](notebooks/cifar_10/) | 10 | 32×32 | 50k | **97.43%** |
| [**CIFAR-100**](notebooks/cifar_100/) | 100 | 32×32 | 50k | **86.25%** |
| [**ImageNet-100**](notebooks/imagenet_100/) | 100 | 224×224 native | 130k | **89.37%** |

Each notebook is self-contained and introduces one new concept — a new architecture, a training technique, or a systematic experiment — building understanding layer by layer.

---

## CIFAR-10 — [4 notebooks](notebooks/cifar_10/)

Starting point: 10 coarse classes, 32×32 images. The goal was establishing a solid baseline progression and understanding the impact of architecture depth and transfer learning before scaling up.

| # | Architecture | Val Acc | Epochs |
|---|-------------|---------|--------|
| 1 | Custom 4-layer CNN | 89.82% | 80 |
| 2 | SimpleResNet (4 stages, custom) | 93.77% | 80 |
| 3 | Custom ResNet-18 | 95.22% | 120 |
| 4 | ResNet-34 (ImageNet pretrained) | **97.43%** | 45 |

> Full development history on the [`v1` branch](../../tree/v1).

---

## CIFAR-100 — [13 notebooks](notebooks/cifar_100/)

The real challenge: 100 fine-grained classes with only **500 training samples per class**. The central question was *where does the ceiling lie for from-scratch training, and what breaks through it?*

| # | Architecture | Strategy | Val Acc | Test Acc |
|---|-------------|----------|---------|----------|
| 1 | SimpleResNet (custom) | Scratch | 75.78% | 74.59% |
| 2 | Custom ResNet-18 | Scratch | 74.15% | 73.42% |
| 3 | ResNet-34 (ImageNet pretrained) | Transfer | 85.50% | 85.03% |
| 4 | ResNet-50 (ImageNet pretrained) | Transfer | **86.64%** | 86.52% |
| 5 | WideResNet-28-4 | Scratch | 78.80% | 78.68% |
| 6 | EfficientNet-B0 (ImageNet pretrained) | Transfer | 86.10% | 85.75% |
| 7 | ResNet-50 | Two-Phase Transfer | 86.28% | 86.25% |
| 8 | Custom EfficientNet-B0 | Scratch | 78.09% | 77.83% |
| 9 | Custom EfficientNet-B3 | Scratch | 78.39% | 77.60% |
| 10 | ResNet-34 — interpolation study | Transfer | 85.77% | 85.89% |

**Key finding:** Scaling from EfficientNet-B0 → B3 from scratch yields almost no gain (~78% both). The bottleneck is data, not model capacity. Transfer learning consistently breaks the ceiling. Notebook 10 systematically compares Bilinear / Bicubic / Lanczos upscaling on CIFAR-100 images before a pretrained ResNet-34.

---

## ImageNet-100 — [4 notebooks](notebooks/imagenet_100/)

Moving to full-resolution (224×224) data to eliminate the CIFAR resolution bottleneck entirely. Dataset: [`ilee0022/ImageNet100`](https://huggingface.co/datasets/ilee0022/ImageNet100) — 100 ImageNet classes, 130k train / 5k val / 13k test.

| # | Architecture | Strategy | Val Acc | Test Acc |
|---|-------------|----------|---------|----------|
| 1 | ResNet-34 (ImageNet pretrained) | Transfer | 85.90% | **89.37%** |
| 2 | ResNet-50 (custom) | Scratch | 85.50% | 88.34% |
| 3 | ResNeXt-50 32×4d (custom) | Scratch | 86.66% | 88.90% |
| 4 | ConvNeXt-Tiny (custom, inline) | Scratch | 82.82% | 86.18% |

---

## Project Structure

```
├── notebooks/
│   ├── cifar_10/           # 4 notebooks — CNN → ResNet-18 → ResNet-34 TL
│   │   └── README.md
│   ├── cifar_100/          # 13 notebooks — scratch ceiling, TL, EfficientNet, interpolation
│   │   └── README.md
│   └── imagenet_100/       # 4 notebooks — full-res, ConvNeXt
│       └── README.md
├── utils/                  # Shared training utilities (all datasets)
│   ├── dataset.py          # Dataloaders & transforms for CIFAR-10/100 & ImageNet-100
│   ├── training.py         # fit(), evaluate(), test_accuracy()
│   ├── callbacks.py        # EarlyStopping, ModelCheckpoint
│   ├── plotting.py         # Training curves, image visualization
│   └── logger.py           # RunLogger — lightweight JSON experiment tracker
├── data/                   # Datasets (gitignored)
├── checkpoint/             # Saved model weights (gitignored)
├── logs/                   # Training run logs (gitignored)
├── requirements.txt
└── README.md
```

---

## Training Pipeline

All notebooks share the same structure:

1. Imports → device setup → data loading
2. Model definition + parameter count
3. Optimizer / Scheduler / AMP `GradScaler` / `ModelCheckpoint` setup
4. `fit()` loop — per-epoch train/val with crash-resume support
5. `restore_best_weights()` → final test evaluation
6. `plot_training_curves()`

**Scheduler convention:**
- From-scratch: `OneCycleLR` stepped **per batch**
- Transfer learning: `CosineAnnealingLR` stepped per epoch

---

## Key Findings

| # | Finding |
|---|---------|
| 1 | **Data, not capacity, is the CIFAR-100 bottleneck.** B0 → B3 EfficientNet from scratch: ~78% both. |
| 2 | **Transfer learning consistently breaks the ceiling** — jumping from ~78% to ~86% on CIFAR-100. |
| 3 | **OneCycleLR must step per batch.** Per-epoch stepping leaves LR stuck near its initial value. |
| 4 | **Interpolation mode matters less than expected.** Lanczos > Bicubic > Bilinear on CIFAR upscaling, but the gaps are small (~0.5% test acc). |

---

## Augmentation

| Dataset | Strategy |
|---------|---------|
| CIFAR-10 | `AutoAugment(CIFAR10)` + `RandomErasing(p=0.25)` |
| CIFAR-100 | `RandAugment(ops=2, mag=9)` + `RandomErasing(p=0.25)` |
| ImageNet-100 | `RandomResizedCrop(224)` + `RandAugment(ops=2, mag=9)` + `RandomErasing(p=0.25)` |

> `AutoAugment(CIFAR10)` is intentionally avoided on CIFAR-100 — its policy is tuned for 10 coarse classes and hurts fine-grained 100-class performance.

---

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.11+, PyTorch 2.x with CUDA, torchvision, matplotlib, numpy  
- ImageNet-100 additionally requires `datasets` and `huggingface-hub`

**Hardware:** Developed on an **NVIDIA RTX 5070 Ti**. `torch.compile` is disabled — Triton is not supported on native Windows.
