# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange

from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule

from opencd.models.utils.builder import ITERACTION_LAYERS


@ITERACTION_LAYERS.register_module()
class ChannelExchange(BaseModule):
    """
    channel exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2, pre_exchange_mask=None):
        N, c, h, w = x1.shape

        if pre_exchange_mask is None:
            exchange_map = torch.arange(c) % self.p == 0
            exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))
        else:
            exchange_mask = pre_exchange_mask

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


@ITERACTION_LAYERS.register_module()
class SpatialExchange(BaseModule):
    """
    spatial exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2


class InteractionMultiFusionModule_Exchange(BaseModule):
    """Pyramid Exchange Module (PEM)
    Parameters:
        channels (int): the number of input channels
        reduction (int): the reduction ratio of the squeeze operation
        p_list (list): the p of the features will be exchanged
    Return the fused pyramid exchange feature map
    """

    def __init__(self, inchannels, p_list=[1/2, 1/4, 1/8, 1/16], mode='spatial'):
        super(InteractionMultiFusionModule_Exchange, self).__init__()
        self.moduleList = nn.ModuleList()
        kernel_size = 1

        if mode == 'spatial':
            kernel_size = 7
            for p in p_list:
                self.moduleList.append(SpatialExchange(p=p))
        elif mode == 'channel':
            for p in p_list:
                self.moduleList.append(ChannelExchange(p=p))
        elif mode == 'mix':
            kernel_size = 7
            for p in p_list:
                self.moduleList.append(SpatialExchange(p=p))
                self.moduleList.append(ChannelExchange(p=p))
        else:
            raise NotImplementedError(f"mode {mode} is not supported.")

        self.attention = EfficientChannelAttention(inchannels * len(p_list))

        self.squeeze = nn.Sequential(
            nn.Conv2d(inchannels * 2 * len(p_list) if mode=='mix' else inchannels * len(p_list),
                      inchannels,
                      kernel_size=kernel_size,
                      padding=kernel_size//2,
                      bias=False),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        outs_x1, outs_x2 = [], []
        for module in self.moduleList:
            out_x1, out_x2 = module(x1, x2)
            outs_x1.append(out_x1)
            outs_x2.append(out_x2)

        outs_x1 = torch.cat(outs_x1, dim=1)
        outs_x2 = torch.cat(outs_x2, dim=1)

        out1, out2 = self.attention(outs_x1) * outs_x1, self.attention(outs_x2) * outs_x2
        out1, out2 = self.squeeze(out1), self.squeeze(out2)

        return out1, out2


@ITERACTION_LAYERS.register_module()
class InteractionMultiFusion_SpatialExchange(BaseModule):
    """
    pyramid spatial exchange
    Args:
        reduction (int): the reduction ratio of the squeeze operation
        p_list (list, optional): p of the features will be exchanged.
            Defaults to [1/2, 1/4, 1/8, 1/16].
    """

    def __init__(self, inchannels, p_list=[1/2, 1/4, 1/8, 1/16]):
        super().__init__()
        self.spatial_imf = InteractionMultiFusionModule_Exchange(inchannels, p_list, mode='spatial')

    def forward(self, x1, x2):
        return self.spatial_imf(x1, x2)


@ITERACTION_LAYERS.register_module()
class InteractionMultiFusion_ChannelExchange(BaseModule):
    """
    pyramid channel exchange
    Args:
        reduction (int): the reduction ratio of the squeeze operation
        p_list (list, optional): p of the features will be exchanged.
            Defaults to [1/2, 1/4, 1/8, 1/16].
    """

    def __init__(self, inchannels, p_list=[1/2, 1/4, 1/8, 1/16]):
        super().__init__()
        self.channel_imf = InteractionMultiFusionModule_Exchange(inchannels, p_list, mode='channel')

    def forward(self, x1, x2):
        return self.channel_imf(x1, x2)


@ITERACTION_LAYERS.register_module()
class InteractionMultiFusion_MixExchange(BaseModule):
    """
    pyramid channel exchange
    Args:
        reduction (int): the reduction ratio of the squeeze operation
        p_list (list, optional): p of the features will be exchanged.
            Defaults to [1/2, 1/4, 1/8, 1/16].
    """

    def __init__(self, inchannels, p_list=[1/2, 1/4, 1/8, 1/16]):
        super().__init__()
        self.mix_imf = InteractionMultiFusionModule_Exchange(inchannels, p_list, mode='mix')

    def forward(self, x1, x2):
        return self.mix_imf(x1, x2)


@ITERACTION_LAYERS.register_module()
class ChannelAttnExchange(ChannelExchange):
    """Channel attention exchange module
    Parameters:
        threshold (float): threshold for the attention decision

    Return the exchanged feature maps
    """

    def __init__(self, threshold=0.5, inchannel=128, reduction=16):
        super().__init__()
        self.two_branches_CR = TwoBranchesCR(inchannel, reduction, out_se=True)
        self.threshold = threshold

    def forward(self, x1, x2):
        d, attn = self.two_branches_CR(x1, x2)

        exchange_mask = attn > self.threshold
        exchange_mask = exchange_mask.squeeze()
        if exchange_mask.dim() == 1:
            exchange_mask = exchange_mask.unsqueeze(0)
        exchange_mask1, exchange_mask2 = torch.chunk(exchange_mask, 2, dim=1)
        exchange_mask = exchange_mask1 | exchange_mask2

        out_x1, out_x2 = super().forward(x1, x2, exchange_mask)

        return d, out_x1, out_x2


@ITERACTION_LAYERS.register_module()
class ChannelNoAttnExchange(ChannelExchange):
    """
    """

    def __init__(self, threshold=0.5, inchannel=128, reduction=16):
        super().__init__()
        self.two_branches_CR = TwoBranchesCR(inchannel, reduction, out_se=True)
        self.threshold = threshold

    def forward(self, x1, x2):
        d, attn = self.two_branches_CR(x1, x2)

        out_x1, out_x2 = super().forward(x1, x2)

        return d, out_x1, out_x2


@ITERACTION_LAYERS.register_module()
class MixExchange(BaseModule):
    """Mix exchange module
    Parameters:
        threshold (float): threshold for the attention decision

    Return the exchanged feature maps
    """

    def __init__(self, p=1/2):
        super().__init__()
        self.channel_exchange = ChannelExchange(p=p)
        self.spatial_exchange = SpatialExchange(p=p)

    def forward(self, x1, x2):
        out_x1, out_x2 = self.channel_exchange(x1, x2)
        out_x1, out_x2 = self.spatial_exchange(out_x1, out_x2)

        return out_x1, out_x2



class SEChannelAttention(BaseModule):
    """SE Channel attention module: https://arxiv.org/abs/1709.01507
    Parameters:
        inchannels (int): input channel dimension

    Return the attention map
    """

    def __init__(self, inchannels, reduction=16):
        super(SEChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=inchannels // reduction, out_channels=inchannels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        weight = self.sigmoid(self.fc(max_out + avg_out))
        return weight


class EfficientChannelAttention(BaseModule):
    def __init__(self, inchannels, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(inchannels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        return out


class SpatialAttention(BaseModule):
    """Spatial attention module: https://arxiv.org/abs/1807.06521

    Return the attention map
    """

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        out = self.sigmoid(self.conv1(torch.cat([max_out, avg_out], dim=1)))
        return out


@ITERACTION_LAYERS.register_module()
class CR(BaseModule):
    """
    diffenence feature interaction
    Args:
    """

    def __init__(self, inchannel, reduction=4):
        super().__init__()
        self.se = SEChannelAttention(inchannel, reduction=4)
        self.spatial_attn = SpatialAttention()

    def forward(self, x1, x2):
        d = torch.abs(x1 - x2)
        f = d * self.se(d)
        d = d * self.spatial_attn(d)
        out = f + d

        return out


@ITERACTION_LAYERS.register_module()
class TwoBranchesCR(BaseModule):
    """
    diffenence feature interaction with two branches
    Args:
    """

    def __init__(self, inchannel, reduction=16, out_se=False):
        super(TwoBranchesCR, self).__init__()
        self.out_se = out_se
        self.conv = nn.Conv2d(int(inchannel * 2), inchannel, kernel_size=1, padding=0)
        self.se = SEChannelAttention(int(inchannel * 2), reduction=reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x1, x2):
        # Addition branch
        x = torch.cat([x1, x2], dim=1)
        se_weights = self.se(x)
        x = x * se_weights
        x = self.conv(x)

        # Difference branch
        d = torch.abs(x1 - x2)
        sp_weights = self.spatial_attn(d)
        d = d * sp_weights

        out = x + d

        if self.out_se:
            return out, se_weights
        else:
            return out


@ITERACTION_LAYERS.register_module()
class IdentityTwoBranchesCR(BaseModule):
    """
    diffenence feature interaction with two branches
    Args:
    """

    def __init__(self, inchannel, out_se=False):
        super(IdentityTwoBranchesCR, self).__init__()
        self.out_se = out_se

    def forward(self, x1, x2):
        # Difference branch
        out = torch.abs(x1 - x2)

        if self.out_se:
            return out, x1, x2
        else:
            return out


@ITERACTION_LAYERS.register_module()
class Aggregation_distribution(BaseModule):
    # Aggregation_Distribution Layer (AD)
    def __init__(self,
                 channels,
                 num_paths=2,
                 attn_channels=None,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_paths = num_paths  # `2` is supported.
        attn_channels = attn_channels or channels // 16
        attn_channels = max(attn_channels, 8)

        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = build_norm_layer(norm_cfg, attn_channels)[1]
        self.act = build_activation_layer(act_cfg)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)
        return x1 * attn1, x2 * attn2


@ITERACTION_LAYERS.register_module()
class TwoIdentity(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x1, x2):
        return x1, x2
