import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import mindspore
from mindspore import Tensor

# 自定义伪影感知模块（非QKV注意力）
class ArtifactInteractionBlock(nn.Cell):
    def __init__(self, channels):
        super(ArtifactInteractionBlock, self).__init__()
        self.local_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        self.local_bn = nn.BatchNorm1d(1)
        self.non_local_fc = nn.Dense(channels, channels)
        self.relu = nn.ReLU()
        self.scale = ms.Tensor(0.1, ms.float32)

    def construct(self, x):
        x_reshape = x.expand_dims(1)  # [B, 1, C]
        local_feat = self.local_conv(x_reshape)
        local_feat = self.local_bn(local_feat)
        local_feat = self.relu(local_feat).squeeze(1)  # [B, C]
        non_local_feat = self.non_local_fc(x)  # [B, C]
        out = x + self.scale * (local_feat + non_local_feat)
        return out


# 改造的 SEBlock（中间插入 ArtifactInteractionBlock）
class SEBlock(nn.Cell):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.fc1 = nn.Dense(channel, channel // reduction)
        self.relu = nn.ReLU()
        self.artifact = ArtifactInteractionBlock(channel // reduction)
        self.fc2 = nn.Dense(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x, (2, 3)).view(b, c)     # [B, C]
        y = self.fc1(y)
        y = self.relu(y)
        y = self.artifact(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


# 辅助函数：通道对齐
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# 辅助函数：标准卷积 + BN + LeakyReLU
def conv_bn(inp, oup, stride=1, leaky=0.1):
    return nn.SequentialCell([
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, pad_mode='same', has_bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(alpha=leaky)
    ])


# 辅助函数：Depthwise Separable 卷积
def conv_dw(inp, oup, stride=1, leaky=0.1):
    return nn.SequentialCell([
        nn.Conv2d(inp, inp, kernel_size=3, stride=stride, pad_mode='same', group=inp, has_bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(alpha=leaky),
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, pad_mode='valid', has_bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(alpha=leaky),
    ])


# 主干网络：轻量化残差网络 + SEBlock（内嵌 Artifact 模块）
class ResidualNN(nn.Cell):
    def __init__(self, alpha=1.0, round_nearest=8):
        super(ResidualNN, self).__init__()
        conv1_out = _make_divisible(32 * alpha, round_nearest)
        self.conv1 = conv_bn(3, conv1_out, stride=2)
        self.se1 = SEBlock(conv1_out)

        stage_channels = [
            (conv1_out, _make_divisible(64 * alpha, round_nearest)),
            (_make_divisible(64 * alpha, round_nearest), _make_divisible(128 * alpha, round_nearest)),
            (_make_divisible(128 * alpha, round_nearest), _make_divisible(128 * alpha, round_nearest)),
            (_make_divisible(128 * alpha, round_nearest), _make_divisible(256 * alpha, round_nearest)),
            (_make_divisible(256 * alpha, round_nearest), _make_divisible(256 * alpha, round_nearest)),
        ]

        self.stage1 = nn.SequentialCell()
        for i, (inp, oup) in enumerate(stage_channels):
            stride = 2 if i in [1, 3] else 1  # downsample at 2nd and 4th block
            self.stage1.append(conv_dw(inp, oup, stride=stride))

        self.se2 = SEBlock(_make_divisible(256 * alpha, round_nearest))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(_make_divisible(256 * alpha, round_nearest), 1)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = self.conv1(x)
        x = self.se1(x)
        x = self.stage1(x)
        x = self.se2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.sigmoid(x)
    


