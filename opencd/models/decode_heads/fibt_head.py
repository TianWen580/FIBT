# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import math
from mmcv.cnn import Conv2d, ConvModule, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, Sequential
from torch.nn import functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from opencd.registry import MODELS
from ..necks.feature_fusion import FeatureFusionNeck


class FDAF(BaseModule):
    """Flow Dual-Alignment Fusion Module.

    Args:
        in_channels (int): Input channels of features.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='IN'),
                 act_cfg=dict(type='GELU')):
        super(FDAF, self).__init__()
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # TODO
        conv_cfg = None
        norm_cfg = dict(type='IN')
        act_cfg = dict(type='GELU')

        kernel_size = 5
        self.flow_make = Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      bias=True, groups=in_channels * 2),
            nn.InstanceNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, 4, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x1, x2, fusion_policy=None):
        """Forward function."""

        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(x1, f1) - x2
        x2_feat = self.warp(x2, f2) - x1

        if fusion_policy == None:
            return x1_feat, x2_feat

        output = FeatureFusionNeck.fusion(x1_feat, x2_feat, fusion_policy)
        return output

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output
    
class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer. \
        Here MixFFN is uesd as projection head of Changer.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


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


def DoubleConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, 1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 1),
        nn.LeakyReLU()
    )


@MODELS.register_module()
class InteractionCDHead(BaseDecodeHead):
    """Interaction change detection head.
    """

    def __init__(self,
                 interaction_cfg=None,
                 is_fdaf=False,
                 upsample_scale=None,
                 channels_list=None,
                 interpolate_mode='bilinear',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        if channels_list is None:
            channels_list = []
        if interaction_cfg is None:
            interaction_cfg = [None, None, None, None]
        if upsample_scale is None:
            upsample_scale = [1, 1, 1, 1]
        self.is_fdaf = is_fdaf
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.CRs = []
        for ia_cfg in interaction_cfg:
            if ia_cfg is None:
                ia_cfg = dict(type='TwoIdentity')
            self.CRs.append(MODELS.build(ia_cfg))
        self.CRs = nn.ModuleList(self.CRs)

        self.d_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(num_inputs):
            if i == 0:
                self.convs.append(nn.ModuleList(None))
                self.d_convs.append(nn.ModuleList(None))
            else:
                self.convs.append(
                    DoubleConv(
                        in_channels=channels_list[i] * 2 + channels_list[i],
                        out_channels=channels_list[i]))
                self.d_convs.append(
                    ConvModule(
                        in_channels=channels_list[i] * 2,
                        out_channels=channels_list[i],
                        kernel_size=1,
                        stride=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.upsamples.append(
                nn.Upsample(
                    scale_factor=upsample_scale[i],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        self.spatial_attn = SpatialAttention()

        self.neck_layer = FDAF(in_channels=self.channels)

        # projection head
        self.discriminator = MixFFN(
            embed_dims=self.channels,
            feedforward_channels=self.channels,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))

    def CRs_forward(self, inputs):
        outs = None
        out_d = None

        for idx in range(len(inputs)):
            conv = self.convs[idx]
            conv_d = self.d_convs[idx]
            if idx == 0:
                x_enc = inputs[-1]
                f1, f2 = self.upsamples[idx](x_enc[0]), self.upsamples[idx](x_enc[1])
                out_d, f1, f2 = self.CRs[idx](f1, f2)
            else:
                x_enc = inputs[len(inputs) - idx - 1]
                f1, f2, out_d = self.upsamples[idx](outs[0]), self.upsamples[idx](outs[1]), self.upsamples[idx](out_d)
                out_d = (out_d + self.CRs[idx](f1, f2)) * 0.5

                f1, f2 = torch.cat([f1, x_enc[0]], dim=1), torch.cat([f2, x_enc[1]], dim=1)
                f1, f2, out_d = conv(f1), conv(f2), conv_d(out_d)
            outs = [f1, f2]

        return outs[0], outs[1], out_d

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        inputs_chunk = []
        for i in inputs:
            f1, f2 = torch.chunk(i, 2, dim=1)
            inputs_chunk.append([f1, f2])

        f1, f2, out_d = self.CRs_forward(inputs_chunk)

        out_d = self.discriminator(out_d)

        out_d = self.cls_seg(out_d)

        if self.is_fdaf:
            out = self.cls_seg(self.neck_layer(f1, f2, 'abs_diff'))
        else:
            out = torch.abs(self.cls_seg(f1) - self.cls_seg(f2))

        out = self.spatial_attn(out) * out + out_d

        return out
