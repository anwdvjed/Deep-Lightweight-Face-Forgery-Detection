# Deep-Lightweight-Face-Forgery-Detection

This repository provides a lightweight PyTorch implementation of a deepfake detector that couples a convolutional image pre-processor with a pure Vision Transformer (ViT) backbone enhanced by Fourier and Discrete Cosine Transform (DCT) features.

## Model overview

The architecture follows three main stages:

1. **CNN preprocessing** – A shallow convolutional stem extracts low-level spatial features while keeping the original resolution intact.
2. **Frequency feature fusion** – Spatial features are concatenated with their Fourier magnitude and 2D DCT counterparts, exposing rich frequency cues that help reveal subtle forgeries.
3. **Transformer classifier** – The fused representation is patch-embedded and processed by a stack of transformer encoder blocks before classification.

The overall design emphasises a small parameter footprint while still capturing both spatial and frequency-domain discrepancies commonly found in forged media.

## Usage

```python
import torch
from models import DeepFakeDetectionModel, DeepFakeDetectionConfig

config = DeepFakeDetectionConfig(
    image_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=2,
)

model = DeepFakeDetectionModel(config)
input_tensor = torch.randn(2, 3, 224, 224)
logits = model(input_tensor)
print(logits.shape)  # torch.Size([2, 2])
```

The model exposes a `from_config` helper to instantiate it using keyword overrides:

```python
model = DeepFakeDetectionModel.from_config(embed_dim=384, depth=8, num_heads=12)
```

## Requirements

- Python 3.9+
- PyTorch 1.10 or later (for `torch.fft` APIs)

Install dependencies via:

```bash
pip install torch
```

## License

This project is provided as-is for research purposes.
