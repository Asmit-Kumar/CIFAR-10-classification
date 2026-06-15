# CIFAR-100 Classification

**100 fine-grained classes · 32×32 images · 60k total (50k train / 10k test) · 500 samples per class**

The central challenge: with only 500 training samples per class, where does from-scratch training actually plateau, and what genuinely breaks through that ceiling?

---

## Results

| # | Notebook | Architecture | Strategy | Params | Val Acc | Test Acc | Epochs |
|---|----------|-------------|----------|--------|---------|----------|--------|
| 1 | `01_simple_resnet.ipynb` | SimpleResNet (custom) | Scratch | ~4.9M | 75.78% | 74.59% | 120 |
| 2 | `02_resnet18.ipynb` | Custom ResNet-18 | Scratch | ~11.2M | 74.15% | 73.42% | 100 |
| 3 | `03_resnet34_transfer.ipynb` | ResNet-34 (ImageNet pretrained) | Transfer | ~21.3M | 85.50% | 85.03% | 120 |
| 4 | `04_resnet50.ipynb` | ResNet-50 (custom) | Scratch | ~23.5M | 79.26% | 79.19% | 90 |
| 5 | `05_resnet50_transfer.ipynb` | ResNet-50 (ImageNet pretrained) | Transfer | ~23.5M | **86.64%** | 86.52% | 60 |
| 6 | `06_wideresnet28.ipynb` | WideResNet-28-4 | Scratch | ~5.9M | 78.80% | 78.68% | 200 |
| 7 | `07_resnext50.ipynb` | ResNeXt-50 (custom) | Scratch | ~23.2M | 77.90% | 78.12% | 90 |
| 8 | `08_efficientnet_b0.ipynb` | Custom EfficientNet-B0-like | Scratch | ~5.3M | 78.09% | 77.83% | 120 |
| 9 | `08b_efficientnet_b0_baseline.ipynb` | EfficientNet-B0 (baseline run) | Scratch | ~5.3M | — | — | — |
| 10 | `09_resnet50_twophase.ipynb` | ResNet-50 (ImageNet pretrained) | Two-Phase TL | ~23.5M | 86.28% | 86.25% | 60 |
| 11 | `10_efficientnet_b3.ipynb` | Custom EfficientNet-B3-like | Scratch | ~10.8M | 78.39% | 77.60% | 120 |
| 12 | `11_efficientnet_b0_transfer.ipynb` | EfficientNet-B0 (ImageNet pretrained) | Transfer | ~4.0M | 86.10% | 85.75% | 40 |
| 13 | `12_interpolation_experiments.ipynb` | ResNet-34 (ImageNet pretrained) | Transfer | ~21.3M | 85.77% | 85.89% | 30 |

---

## Interpolation Study (notebook 12)

Does the choice of interpolation mode matter for downstream classification accuracy when upscaling 32×32 images to 224×224?

This study uses a controlled two-phase ResNet-34 fine-tuning setup (10 epoch head warm-up + 20 epoch full fine-tune) to isolate the effect of interpolation mode:

| Mode | Phase 1 Val | Best Val | Test Acc |
|------|------------|----------|----------|
| Bilinear | 62.22% | 85.62% | 85.37% |
| Bicubic | 61.47% | 85.69% | 85.78% |
| **Lanczos** | 62.11% | **85.77%** | **85.89%** |

Lanczos wins, but the gaps are marginal (~0.5% test acc across all three). The takeaway: for standard upscaling, all three modes are roughly equivalent — switching from Bilinear to Lanczos is free, but don't expect a breakthrough.

### On Learned Super-Resolution

Before settling on interpolation comparison, I tested various CNN-based super-resolution models to see if a learned upscaler could provide better inputs to the pretrained backbone than standard interpolation (I may push the experiment repo for this later).

The problem: CIFAR images require a **7× upscaling factor** (32→224). I found that I was asking a little too much for upscaling: 7x enhancements gave almost the exact same downstream accuracy results as bicubic interpolation. The hallucinated textures didn't help the ImageNet classifier, so I didn't go further on that path.

---

## The From-Scratch Ceiling

The data-limited regime is brutal. These experiments confirm it empirically:

- **SimpleResNet → ResNet-18 → WRN-28-4 → EfficientNet-B0 → EfficientNet-B3 (all scratch):** Stuck at **~74–79%**. Scaling capacity does almost nothing — scaling from EfficientNet-B0 (~5.3M params) to B3 (~10.8M params) yields only +0.3% accuracy.
- The bottleneck is the **500 samples per class**, not the model. Heavier augmentation (already applied) and longer training don't move the needle either.
- **Transfer learning immediately jumps to ~85–86%** — the first pretrained model (ResNet-34) beats the best from-scratch result by over 6 points.

---

## EfficientNet-B0 Scheduler Experiments (notebook 08)

Systematic LR schedule ablations to understand what actually drives performance for this class of model on CIFAR-100.

All runs: `SGD + MixUp(α=0.2) + label_smoothing=0.1`

| Run | Scheduler | Notes | Best Val Acc |
|-----|-----------|-------|-------------|
| v1 | `OneCycleLR` (max_lr=0.1) | 120 epochs — **reference** | **78.09%** |
| v2 | `OneCycleLR` (max_lr=0.04) | Longer warmup, lower peak | 74.28% |
| v3 | `CosineAnnealingLR` | With MixUp | 75.36% |
| v4 | `OneCycleLR` | **Bug:** scheduler stepped per epoch instead of per batch | 73.46% |
| v5 | `CosineAnnealingLR` | 150 epochs, no MixUp | 77.31% |

**Key bug caught:** `OneCycleLR` *must* be stepped per batch (`step_scheduler_per_batch=True`). Stepping per epoch keeps the LR frozen at its initial value (~0.01) for the entire run — run v4 confirmed this is a silent killer.

---

## Augmentation

| Phase | Pipeline |
|-------|---------|
| Train | `RandAugment(ops=2, mag=9)` → `RandomCrop(32, pad=4)` → `RandomHorizontalFlip` → `RandomErasing(p=0.25)` |
| Val/Test | `ToTensor()` → `Normalize` |

`AutoAugment(CIFAR10)` is intentionally **not used** — its policy is optimised for 10 coarse classes and consistently hurts performance on CIFAR-100's fine-grained 100-class structure. `RandAugment` with no class-specific prior generalises better here.
