import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape)
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
                             f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1).contiguous(), self.normalized_shape,
            self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNextEncoder(nn.Module):
    def __init__(self,
                 input_channels=3,
                 blocks_per_stage=[3, 3, 9, 3],
                 channels_per_stage=[96, 192, 384, 768],
                 drop_path_rate=0.):
        super(ConvNextEncoder, self).__init__()
        self.num_stages = 4
        self.channels_per_stage = channels_per_stage
        self.blocks_per_stage = blocks_per_stage
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels_per_stage[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(channels_per_stage[0])
        )
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(blocks_per_stage))]

        curr = 0
        self.stage_1 = nn.Sequential(
            *[Block(dim=channels_per_stage[0], drop_path=dp_rates[curr + j]) for j in range(blocks_per_stage[0])])
        self.downsample_1 = nn.Sequential(
            LayerNorm(channels_per_stage[0], eps=1e-6),
            nn.Conv2d(channels_per_stage[0], channels_per_stage[1], kernel_size=2, stride=2),
        )
        curr += blocks_per_stage[0]

        self.stage_2 = nn.Sequential(
            *[Block(dim=channels_per_stage[1], drop_path=dp_rates[curr + j]) for j in range(blocks_per_stage[1])])
        self.downsample_2 = nn.Sequential(
            LayerNorm(channels_per_stage[1], eps=1e-6),
            nn.Conv2d(channels_per_stage[1], channels_per_stage[2], kernel_size=2, stride=2),
        )
        curr += blocks_per_stage[1]

        self.stage_3 = nn.Sequential(
            *[Block(dim=channels_per_stage[2], drop_path=dp_rates[curr + j]) for j in range(blocks_per_stage[2])])
        self.downsample_3 = nn.Sequential(
            LayerNorm(channels_per_stage[2], eps=1e-6),
            nn.Conv2d(channels_per_stage[2], channels_per_stage[3], kernel_size=2, stride=2),
        )
        curr += blocks_per_stage[2]

        self.stage_4 = nn.Sequential(
            *[Block(dim=channels_per_stage[3], drop_path=dp_rates[curr + j]) for j in range(blocks_per_stage[3])])
        self.downsample_4 = nn.Sequential(
            LayerNorm(channels_per_stage[3], eps=1e-6),
            nn.Conv2d(channels_per_stage[3], channels_per_stage[3], kernel_size=2, stride=2),
        )
        curr += blocks_per_stage[3]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        pass

    def forward_first_conv(self, x):
        x = self.stem(x)
        return x

    def forward_stage_1(self, x):
        feat = self.stage_1(x)
        out = self.downsample_1(feat)
        return out, feat

    def forward_stage_2(self, x):
        feat = self.stage_2(x)
        out = self.downsample_2(feat)
        return out, feat

    def forward_stage_3(self, x):
        feat = self.stage_3(x)
        out = self.downsample_3(feat)
        return out, feat

    def forward_stage_4(self, x):
        out = self.stage_4(x)
        out = self.downsample_4(out)
        return out