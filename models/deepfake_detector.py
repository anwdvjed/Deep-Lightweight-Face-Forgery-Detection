"""Vision Transformer based deepfake detection model with frequency enhancements."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def _dct_1d(x: Tensor, dim: int) -> Tensor:
    """Apply a type-II Discrete Cosine Transform along a dimension."""
    return torch.fft.dct(x, type=2, n=None, dim=dim, norm="ortho")


def dct_2d(x: Tensor) -> Tensor:
    """Compute a 2D DCT on the spatial dimensions of a tensor."""
    # Apply DCT along width and height separately.
    x = _dct_1d(x, dim=-1)
    x = _dct_1d(x, dim=-2)
    return x


class CNNPreprocessor(nn.Module):
    """A lightweight CNN feature extractor for image pre-processing."""

    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class FrequencyMixer(nn.Module):
    """Combine spatial, Fourier, and DCT features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.out_channels = channels * 3

    def forward(self, x: Tensor) -> Tensor:
        spatial = x
        fourier = torch.fft.fft2(x, norm="ortho")
        fourier = torch.abs(fourier)
        cosine = dct_2d(x)
        return torch.cat([spatial, fourier, cosine], dim=1)


class PatchEmbedding(nn.Module):
    """Patch embedding implemented as a strided convolution."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        patches = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        b, c, h, w = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return self.norm(patches)


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return self.proj_drop(out)


class MLP(nn.Module):
    """Transformer MLP block."""

    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class DeepFakeDetectionConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    cnn_hidden_channels: int = 32
    cnn_out_channels: int = 64
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_classes: int = 2


class DeepFakeDetectionModel(nn.Module):
    """Deepfake detector that fuses CNN preprocessing with a frequency-aware ViT."""

    def __init__(self, config: Optional[DeepFakeDetectionConfig] = None) -> None:
        super().__init__()
        self.config = config or DeepFakeDetectionConfig()
        cfg = self.config
        self.preprocessor = CNNPreprocessor(cfg.in_channels, cfg.cnn_hidden_channels, cfg.cnn_out_channels)
        self.frequency_mixer = FrequencyMixer(cfg.cnn_out_channels)
        self.patch_embed = PatchEmbedding(self.frequency_mixer.out_channels, cfg.embed_dim, cfg.patch_size)

        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.preprocessor(x)
        x = self.frequency_mixer(x)
        x = self.patch_embed(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        return self.head(features)

    @classmethod
    def from_config(cls, **kwargs: object) -> "DeepFakeDetectionModel":
        """Create a model from keyword configuration overrides."""
        return cls(DeepFakeDetectionConfig(**kwargs))


__all__ = [
    "DeepFakeDetectionConfig",
    "DeepFakeDetectionModel",
]
