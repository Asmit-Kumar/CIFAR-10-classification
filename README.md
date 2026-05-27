# CIFAR-10 / CIFAR-100 Classification with PyTorch

A collection of deep learning experiments progressing from a plain CNN to custom
EfficientNet-style architectures, evaluated on both **CIFAR-10** (10 classes) and
**CIFAR-100** (100 classes). The primary goal is learning — each notebook introduces
a new concept, architectural improvement, or training technique.

---

## CIFAR-10 Results

| # | Notebook | Architecture | Params | Val Acc | Epochs |
|---|----------|-------------|--------|---------|--------|
| 1 | `01_plain_cnn.ipynb` | Custom 4-layer CNN | ~958k | 89.82% | 80 |
| 2 | `02_simple_resnet.ipynb` | SimpleResNet (4 stages) | ~4.9M | 93.77% | 80 |
| 3 | `03_resnet18.ipynb` | Custom ResNet-18 | ~11.2M | 95.22% | 120 |
| 4 | `04_resnet34_transfer.ipynb` | ResNet-34 (ImageNet TL) | ~21.3M | 97.43% | 45 |

---

## CIFAR-100 Results

| # | Notebook | Architecture | Strategy | Params | Val Acc | Test Acc | Epochs |
|---|----------|-------------|----------|--------|---------|----------|--------|
| 1 | `02_simple_resnet_100.ipynb` | SimpleResNet | From scratch | ~4.9M | 75.78% | 74.59% | 120 |
| 2 | `03_resnet18_100.ipynb` | ResNet-18 | From scratch | ~11.2M | 74.15% | 73.42% | 100 |
| 3 | `04_resnet34_transfer_100.ipynb` | ResNet-34 (ImageNet TL) | Transfer | ~21.3M | 85.50% | 85.03% | 120 |
| 4 | `05_resnet50_transfer_100.ipynb` | ResNet-50 (ImageNet TL) | Transfer | ~23.5M | **86.64%**| TBD | 60 |
| 5 | `05_wideresnet28_100.ipynb` | WideResNet-28-4 | From scratch | ~5.9M | 78.80% | 78.68% | 200 |
| 6 | `06_efficientnet_100.ipynb` | Custom EfficientNet-B0-like | From scratch | ~5.3M | 78.09% | 77.83% | 120 |
| 7 | `06_resnet50_twophase_100.ipynb` | ResNet-50 (ImageNet TL) | Two-Phase TL | ~23.5M | 86.28% | 86.25% | 60 |
| 8 | `07_efficientnet-b3_100.ipynb` | Custom EfficientNet-B3-like | From scratch | ~10.8M | 78.39% | 77.60% | 120 |
| 9 | `07_efficientnet_b0_transfer_100.ipynb` | EfficientNet-B0 (ImageNet TL) | Transfer | ~4.0M | 86.10% | 85.75% | 40 |

### EfficientNet-B0 LR Experiments (notebook 06)

All runs: `SGD + OneCycleLR (max_lr=0.1, pct_start=10%) + MixUp(α=0.2) + label_smoothing=0.1`

| Run | Notes | Best Val Acc |
|-----|-------|-------------|
| v1 | max_lr=0.1, 120 ep | 78.09% ✓ |
| v2 | max_lr=0.04, longer warmup | 74.28% |
| v3 | CosineAnnealingLR + MixUp | 75.36% |
| v4 | OneCycleLR, scheduler bug (stepped/epoch not /batch) | 73.46% |
| v5 | 150 epochs (CosineAnnealingLR, no MixUp) | 77.31% |

**Key finding:** OneCycleLR **must** be stepped per batch (`step_scheduler_per_batch=True`).
Stepping per epoch leaves the LR stuck at the initial value (~0.01) for the entire run.

### From-Scratch Ceiling on CIFAR-100

CIFAR-100 has only **500 samples per class**. The experiments confirm a hard ceiling:

- **EfficientNet-B0 → B3 (scratch): both ~78%** — scaling the architecture provided
  almost no gain, confirming the bottleneck is data, not model capacity.
- **Transfer learning breaks the ceiling** — ResNet-34 (ImageNet) reaches ~84%
  because pretrained features learned from 1.2M+ images transfer to fine-grained
  CIFAR-100 classes.
- **Augmentation pipeline** (RandAugment + RandomErasing) is already applied in all
  CIFAR-100 runs — increasing augmentation strength did not push past 78%.

---

## Project Structure

```
CIFAR_10/
├── notebooks/          # Progressively complex experiments
│   ├── 01_plain_cnn.ipynb                # CIFAR-10: baseline CNN
│   ├── 02_simple_resnet.ipynb            # CIFAR-10: SimpleResNet
│   ├── 02_simple_resnet_100.ipynb        # CIFAR-100: SimpleResNet
│   ├── 03_resnet18.ipynb                 # CIFAR-10: custom ResNet-18
│   ├── 03_resnet18_100.ipynb             # CIFAR-100: ResNet-18
│   ├── 04_resnet34_transfer.ipynb        # CIFAR-10: ResNet-34 (TL)
│   ├── 04_resnet34_transfer_100.ipynb    # CIFAR-100: ResNet-34 (TL)
│   ├── 05_resnet50_transfer_100.ipynb    # CIFAR-100: ResNet-50 (TL)
│   ├── 05_wideresnet28_100.ipynb         # CIFAR-100: WideResNet-28-4
│   ├── 06_efficientnet_100.ipynb         # CIFAR-100: custom EfficientNet-B0-like
│   ├── 06_resnet50_twophase_100.ipynb    # CIFAR-100: ResNet-50 two-phase TL
│   ├── 07_efficientnet-b3_100.ipynb      # CIFAR-100: custom EfficientNet-B3-like
│   └── 07_efficientnet_b0_transfer_100.ipynb  # CIFAR-100: EfficientNet-B0 (TL)
├── utils/              # Reusable shared utilities
│   ├── __init__.py        # Package exports
│   ├── callbacks.py       # EarlyStopping, ModelCheckpoint
│   ├── dataset.py         # Dataloaders & transforms (CIFAR-10 & CIFAR-100)
│   ├── training.py        # fit(), evaluate(), test_accuracy()
│   ├── plotting.py        # plot_training_curves(), show_sample_batch()
│   └── logger.py          # JSON experiment tracker (RunLogger)
├── data/               # Downloaded datasets (CIFAR-10 & CIFAR-100)
├── checkpoint/         # Saved model weights (.pth)
├── archive/            # Legacy notebooks and backups
├── requirements.txt
└── README.md
```

---

## Augmentation Strategy

| Dataset | Augmentation |
|---------|-------------|
| CIFAR-10 | `AutoAugment(CIFAR10)` + `RandomErasing(p=0.25)` |
| CIFAR-100 | `RandAugment(ops=2, mag=9)` + `RandomErasing(p=0.25)` |

`AutoAugment(CIFAR10)` is intentionally **not** used for CIFAR-100 — its policy is tuned
for 10 coarse classes and degrades on fine-grained 100-class structure.

---

## Training Pipeline

All notebooks follow the same flow:
1. Imports → Device setup → Data loading
2. Model definition + parameter count
3. Optimizer / Scheduler / AMP (`GradScaler`) / Checkpoint setup
4. `fit()` training loop with per-epoch validation
5. `restore_best_weights()` → Test evaluation
6. `plot_training_curves()`

**Scheduler policy:**
- From-scratch models: `OneCycleLR` — stepped **per batch**
- Transfer learning: `CosineAnnealingLR` — stepped per epoch

---

## Requirements

- Python 3.11+
- PyTorch 2.x (CUDA)
- torchvision, matplotlib, numpy

```bash
pip install -r requirements.txt
```

## Hardware

Developed and tested on **NVIDIA RTX 5070 Ti** with CUDA.  
`torch.compile` is disabled — Triton is unsupported on native Windows.
