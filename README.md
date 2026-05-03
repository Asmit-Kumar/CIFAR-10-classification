# CIFAR-10 Image Classification with PyTorch

A collection of deep learning experiments on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, progressing from a plain CNN to transfer learning with a pre-trained ResNet-34.

## Models

| # | Notebook | Architecture | Epochs | Key Idea |
|---|----------|-------------|--------|----------|
| 1 | `01_plain_cnn.ipynb` | Custom 4-layer CNN | 80 | Baseline — no skip connections |
| 2 | `02_simple_resnet.ipynb` | SimpleResBlock × 4 stages | 80 | Introduces residual connections |
| 3 | `03_resnet18.ipynb` | Custom ResNet-18 | 120 | Full ResNet with `_make_layer` |
| 4 | `04_resnet34_transfer.ipynb` | Pre-trained ResNet-34 | 45 | Transfer learning from ImageNet |

## Project Structure

```
CIFAR_10/
├── data/                      # CIFAR-10 dataset (auto-downloaded)
├── checkpoint/                # Saved model weights
├── notebooks/                 # Experiment notebooks
│   ├── 01_plain_cnn.ipynb
│   ├── 02_simple_resnet.ipynb
│   ├── 03_resnet18.ipynb
│   └── 04_resnet34_transfer.ipynb
├── utils/                     # Shared utilities
│   ├── callbacks.py           # EarlyStopping, ModelCheckpoint
│   ├── dataset.py             # Data loading & transforms
│   ├── training.py            # Training loops & evaluation
│   └── plotting.py            # Visualization helpers
├── archive/                   # Old notebooks (preserved)
├── requirements.txt
└── README.md
```

## Shared Utilities

All notebooks import from the `utils/` package:

- **`dataset.py`** — `get_cifar10_dataloaders()` with configurable augmentation, image size (32 or 224), and train/val splitting
- **`training.py`** — `fit()` for the full training loop, `evaluate()`, `test_accuracy()` with per-class breakdown
- **`plotting.py`** — `plot_training_curves()`, `imshow()`, `show_sample_batch()`
- **`callbacks.py`** — `ModelCheckpoint` (crash protection + best model saving), `EarlyStopping`

## Training Pipeline

All notebooks follow the same flow:
1. Imports → Device setup → Data loading
2. Model definition → Parameter summary
3. Hyperparameters (SGD + OneCycleLR + AMP)
4. Training loop with validation
5. Best weights restoration → Test evaluation (overall + per-class)
6. Training curves plot

## Requirements

- Python 3.12+
- PyTorch 2.11+ (CUDA)
- torchvision, matplotlib, numpy

```bash
pip install -r requirements.txt
```

## Hardware

Developed and tested on an **NVIDIA RTX 5070 Ti** with CUDA.


# Note
The archieve contains all the previous nbs created by me; the Code base is organised by AI
