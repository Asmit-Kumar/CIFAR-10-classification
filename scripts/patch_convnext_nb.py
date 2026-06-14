"""
Patch 04_convnext_tiny.ipynb:
1. Replace `from convnext import ConvNextT` with the full inline model definition
2. Update sys.path.append('..') to sys.path.append('../..') for new directory structure
"""

import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "imagenet_100" / "04_convnext_tiny.ipynb"

# The full ConvNeXt model code to inline
CONVNEXT_CODE = '''import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticDepth(nn.Module):
    """Drop Path (Stochastic Depth) regularization per block."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob

class LayerNorm2d(nn.Module):
    """
    LayerNorm that supports (N, C, H, W) inputs.
    We permute to (N, H, W, C), apply standard nn.LayerNorm, and permute back.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class ConvNeXtBlock(nn.Module):

    def __init__(self, dim, layer_scale_init_value=1e-6, drop_path=0.,):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        
        x = input + self.drop_path(x)
        return x


class ConvNextT(nn.Module):
    """
    ConvNeXt-Tiny for ImageNet (224x224).
    Standard architecture: 
    - Stem: 4x4 conv, stride 4
    - Stages: [3, 3, 9, 3] blocks
    - Dims: [96, 192, 384, 768]
    """
    def __init__(self,drop_path_rate = 0.1, num_classes: int=100) -> None:
        super().__init__()
        
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]

        
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
            
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
            
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.fc = nn.Linear(dims[-1], num_classes)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.fc(x)
        return x
'''

def patch():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])

        # Fix sys.path for new directory depth (notebooks/imagenet_100/ is 2 levels deep)
        if "sys.path.append('..')" in source:
            cell['source'] = [line.replace("sys.path.append('..')", "sys.path.append('../..')") for line in cell['source']]

        # Replace the convnext import with inline definition
        if 'from convnext import ConvNextT' in source:
            new_lines = []
            for line in cell['source']:
                if 'from convnext import ConvNextT' in line:
                    # Skip this import line - we'll add the model in a new cell
                    continue
                new_lines.append(line)
            cell['source'] = new_lines

    # Find the import cell and insert a new model definition cell right after it
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if "sys.path.append('../..')" in source:
            import_cell_idx = i
            break

    # Create the model definition cell
    model_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in CONVNEXT_CODE.strip().split('\n')]
    }

    # Also create a markdown header cell for the model
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## ConvNeXt-Tiny Model Definition\n",
            "\n",
            "Custom implementation with depthwise 7×7 convolutions, GELU, LayerNorm, layer scale, and stochastic depth."
        ]
    }

    # Insert after the import cell
    nb['cells'].insert(import_cell_idx + 1, md_cell)
    nb['cells'].insert(import_cell_idx + 2, model_cell)

    # Also update any markdown references to ../convnext.py
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            cell['source'] = [
                line.replace('../convnext.py', 'this notebook')
                    .replace('`../convnext.py`', 'this notebook')
                    .replace('from `this notebook`', 'inline in this notebook')
                for line in cell['source']
            ]

    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')

    print(f"[OK] Patched {NB_PATH}")
    print(f"   - sys.path updated to '../..'")
    print(f"   - ConvNeXt model definition inlined")
    print(f"   - Removed external import")


if __name__ == '__main__':
    patch()
