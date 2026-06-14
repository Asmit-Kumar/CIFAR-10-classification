# CIFAR-10 Classification

**10 classes · 32×32 images · 60k total (50k train / 10k test)**

The starting point of this project. The goal here wasn't just to get good numbers — it was to build every component from scratch and understand each step: data pipelines, custom architectures, residual connections, and eventually transfer learning.

---

## Progression

| # | Notebook | Architecture | Params | Val Acc | Test Acc | Epochs |
|---|----------|-------------|--------|---------|----------|--------|
| 1 | `01_plain_cnn.ipynb` | Custom 4-layer CNN | ~958k | 89.82% | — | 80 |
| 2 | `02_simple_resnet.ipynb` | SimpleResNet (custom 4-stage) | ~4.9M | 93.77% | — | 80 |
| 3 | `03_resnet18.ipynb` | Custom ResNet-18 | ~11.2M | 95.22% | — | 120 |
| 4 | `04_resnet34_transfer.ipynb` | ResNet-34 (ImageNet pretrained) | ~21.3M | **97.43%** | — | 45 |

Each notebook introduces exactly one change over the previous — making it easy to attribute the accuracy delta to the specific improvement.

---

## What Each Notebook Covers

- **`01_plain_cnn`** — Baseline: 4 conv blocks with BatchNorm + MaxPool. Establishes the data pipeline and training loop that everything else builds on.
- **`02_simple_resnet`** — Custom SimpleResNet with residual connections, SE attention blocks, and label smoothing. First real jump in accuracy.
- **`03_resnet18`** — Custom ResNet-18 following the original paper's architecture. Validates that the implementation matches expected performance.
- **`04_resnet34_transfer`** — Pretrained ResNet-34 fine-tuned with two-phase transfer learning (head warm-up → full fine-tune). Demonstrates the power of ImageNet features for small datasets.

---

## Augmentation

| Phase | Pipeline |
|-------|---------|
| Train | `AutoAugment(CIFAR10)` → `RandomCrop(32, pad=4)` → `RandomHorizontalFlip` → `RandomErasing(p=0.25)` |
| Val/Test | `ToTensor()` → `Normalize` |

> `AutoAugment(CIFAR10)` uses a policy learned specifically on CIFAR-10 and provides meaningful accuracy gains over manual augmentation.

---

> **Note:** For the full development history, earlier iterations, and exploratory notebooks from when this project started, see the [`v1` branch](../../tree/v1).
