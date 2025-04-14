import math
from abc import abstractmethod
from dataclasses import dataclass
from numbers import Number

import torch as th
import torch.nn.functional as F
from choices import *
from config_base import BaseConfig
from torch import nn

from .nn import (avg_pool_nd, conv_nd, linear, normalization,
                 timestep_embedding, torch_checkpoint, zero_module)


class ScaleAt(Enum):
    after_norm = 'afternorm'


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb=None, cond=None, lateral=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb=None, cond=None, lateral=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb=emb, cond=cond, lateral=lateral)
            else:
                x = layer(x)
        return x


@dataclass
class ResBlockConfig(BaseConfig):
    channels: int
    emb_channels: int
    dropout: float
    out_channels: int = None
    # condition the resblock with time (and encoder's output)
    use_condition: bool = True
    # whether to use 3x3 conv for skip path when the channels aren't matched
    use_conv: bool = False
    # dimension of conv (always 2 = 2d)
    dims: int = 2
    # gradient checkpoint
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False
    # whether to condition with both time & encoder's output
    two_cond: bool = False
    # number of encoders' output channels
    cond_emb_channels: int = None
    # suggest: False
    has_lateral: bool = False
    lateral_channels: int = None
    # whether to init the convolution with zero weights
    # this is default from BeatGANs and seems to help learning
    use_zero_module: bool = True

    def __post_init__(self):
        self.out_channels = self.out_channels or self.channels
        self.cond_emb_channels = self.cond_emb_channels or self.emb_channels

    def make_model(self):
        return ResBlock(self)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    total layers:
        in_layers
        - norm
        - act
        - conv
        out_layers
        - norm
        - act
        - dropout
        - conv
    """
    def __init__(self, conf: ResBlockConfig):
        super().__init__()
        self.conf = conf

        #############################
        # IN LAYERS
        #############################
        assert conf.lateral_channels is None
        layers = [
            normalization(conf.channels),
            nn.SiLU(),
            conv_nd(conf.dims, conf.channels, conf.out_channels, 3, padding=1)
        ]
        self.in_layers = nn.Sequential(*layers)

        self.updown = conf.up or conf.down

        if conf.up:
            self.h_upd = Upsample(conf.channels, False, conf.dims)
            self.x_upd = Upsample(conf.channels, False, conf.dims)
        elif conf.down:
            self.h_upd = Downsample(conf.channels, False, conf.dims)
            self.x_upd = Downsample(conf.channels, False, conf.dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        #############################
        # OUT LAYERS
        #############################
        conv = conv_nd(conf.dims,
                       conf.out_channels,
                       conf.out_channels,
                       3,
                       padding=1)
        if conf.use_zero_module:
            conv = zero_module(conv)

        self.out_layers = nn.Sequential(
            normalization(conf.out_channels),
            nn.SiLU(),
            nn.Dropout(p=conf.dropout),
            conv,
        )

        #############################
        # SKIP LAYERS
        #############################
        if conf.out_channels == conf.channels:
            self.skip_connection = nn.Identity()
        else:
            if conf.use_conv:
                kernel_size = 3
                padding = 1
            else:
                kernel_size = 1
                padding = 0

            self.skip_connection = conv_nd(conf.dims,
                                           conf.channels,
                                           conf.out_channels,
                                           kernel_size,
                                           padding=padding)

    def forward(self, x, emb=None, cond=None, lateral=None):
        """
        Apply the block to a Tensor, with optional lateral connection.

        Args:
            x: input
            lateral: lateral connection from the encoder (if applicable)
        """
        return torch_checkpoint(self._forward, (x, lateral), self.conf.use_checkpoint)

    def _forward(self, x, lateral=None):
        if self.conf.has_lateral:
            assert lateral is not None
            x = th.cat([x, lateral], dim=1)

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        h = self.out_layers(h)
        return self.skip_connection(x) + h


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims,
                                self.channels,
                                self.out_channels,
                                3,
                                padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                              mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims,
                              self.channels,
                              self.out_channels,
                              3,
                              stride=stride,
                              padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return torch_checkpoint(self._forward, (x, ), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch,
                                                                       dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale,
            k * scale)  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight,
                      v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]
