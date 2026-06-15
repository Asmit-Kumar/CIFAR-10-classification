# ImageNet-100 Classification

**100 ImageNet classes · 224×224 native resolution · ~148k images (130k train / 5k val / 13k test)**

Dataset: [`ilee0022/ImageNet100`](https://huggingface.co/datasets/ilee0022/ImageNet100) — loaded via Hugging Face `datasets` with a custom PyTorch wrapper to avoid memory leaks from PyArrow multiprocessing on Windows.

Moving beyond CIFAR's 32×32 constraint to real-world high-resolution data. These experiments cover full fine-tuning, from-scratch training of modern architectures, and a systematic study of upscaling strategies.

---

## Results

| # | Notebook | Architecture | Strategy | Params | Val Acc | Test Acc | Epochs |
|---|----------|-------------|----------|--------|---------|----------|--------|
| 1 | `01_resnet34_transfer.ipynb` | ResNet-34 (ImageNet pretrained) | Transfer | ~21.3M | 85.90% | **89.37%** | 30 |
| 2 | `02_resnet50.ipynb` | ResNet-50 (custom) | Scratch | ~23.7M | 85.50% | 88.34% | 90 |
| 3 | `03_resnext50.ipynb` | ResNeXt-50 32×4d (custom) | Scratch | ~23.2M | 86.66% | 88.90% | 90 |
| 4 | `04_convnext_tiny.ipynb` | ConvNeXt-Tiny (custom, inline) | Scratch | ~27.9M | 82.82% | 86.18% | 90 |

---

## ConvNeXt-Tiny (notebook 04)

A from-scratch implementation of ConvNeXt-Tiny trained end-to-end on ImageNet-100. The model definition is inlined directly into the notebook.

**Architecture:**
- Patch stem: 4×4 conv, stride 4
- 4 stages with depths [3, 3, 9, 3], dims [96, 192, 384, 768]
- Depthwise 7×7 conv → channel-last LayerNorm → GELU MLP (4× expand)
- Layer Scale (`γ = 1e-6`) + Stochastic Depth (drop rate 0.1)

At 90 epochs from scratch, ConvNeXt-Tiny reaches 86.18% test accuracy — competitive with pretrained ResNet-50/34 at 88–89%, and without any pretrained weights.

---

## Augmentation

| Phase | Pipeline |
|-------|---------|
| Train | `RandomResizedCrop(224)` → `RandomHorizontalFlip` → `RandAugment(ops=2, mag=9)` → `RandomErasing(p=0.25)` → `Normalize(ImageNet)` |
| Val/Test | `Resize(256)` → `CenterCrop(224)` → `Normalize(ImageNet)` |
