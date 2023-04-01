import copy
from functools import partial
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import List, Optional, Tuple
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange, Reduce
from torch import einsum

from gigagan.open_clip import OpenClipAdapter
from gigagan.utils import (conv2d_3x3, default, exists, is_power_of_two,
                           leaky_relu, safe_unshift)


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_same_padding(size: int, kernel_size: int, dilation: int,
                     stride: int) -> int:
    """Calculate padding size to keep output feature map size the same.

    Args:
        size (int): Input feature map size
        kernel_size (int): Convolution kernel size
        dilation (int): Convolution dilation
        stride (int): Convolution stride

    Returns:
        (int): Padding size
    """
    padding = (((size - 1) * stride) - size + (dilation *
                                               (kernel_size - 1)) + 1) // 2
    return padding


class ChannelRMSNorm(nn.Module):
    """Channel-wise Root Mean Squared (RMS) Normalization module.

    This module normalizes the input tensor along the channel dimension using the RMS value of each channel.
    The output tensor is then scaled by a learnable parameter and a scaling factor.

    Args:
        dim (int): The number of channels in the input tensor.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Calculate the scaling factor based on the number of channels
        self.scale = dim**0.5
        # Initialize the scaling parameter as a learnable parameter
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ChannelRMSNorm module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, dim, height, width).

        Returns:
            torch.Tensor: The normalized and scaled tensor of shape (batch_size, dim, height, width).
        """
        # Normalize the input tensor along the channel dimension
        normed = F.normalize(x, dim=1)
        # Scale the normalized tensor by the scaling factor and the learnable scaling parameter
        return normed * self.scale * self.gamma


class RMSNorm(nn.Module):
    """Root Mean Squared (RMS) Normalization module.

    This module normalizes the input tensor along the last dimension using the RMS value of each element.
    The output tensor is then scaled by a learnable parameter and a scaling factor.

    Args:
        dim (int): The size of the last dimension in the input tensor.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Calculate the scaling factor based on the size of the last dimension
        self.scale = dim**0.5
        # Initialize the scaling parameter as a learnable parameter
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, ..., dim).

        Returns:
            torch.Tensor: The normalized and scaled tensor of shape (batch_size, ..., dim).
        """
        # Normalize the input tensor along the last dimension
        normed = F.normalize(x, dim=-1)
        # Scale the normalized tensor by the scaling factor and the learnable scaling parameter
        return normed * self.scale * self.gamma


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        conv = nn.Conv2d(dim, dim * 4, 1)

        self.net = nn.Sequential(conv, nn.SiLU(), nn.PixelShuffle(2))

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


def Downsample(dim):
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1=2, s2=2),
        nn.Conv2d(dim * 4, dim, 1))


# skip layer excitation


def SqueezeExcite(dim, dim_out, reduction=4, dim_min=32):
    dim_hidden = max(dim_out // reduction, dim_min)

    return nn.Sequential(Reduce('b c h w -> b c', 'mean'),
                         nn.Linear(dim, dim_hidden), nn.SiLU(),
                         nn.Linear(dim_hidden, dim_out), nn.Sigmoid(),
                         Rearrange('b c -> b c 1 1'))


class AdaptiveConv2DMod(nn.Module):
    def __init__(self,
                 dim: int,
                 dim_out: int,
                 dim_embed: int,
                 kernel: int,
                 demod: bool = True,
                 stride: int = 1,
                 dilation: int = 1,
                 eps: float = 1e-8,
                 num_conv_kernels: Optional[int] = 1):
        """Adaptive Convolutional Layer with Modulated Weights, as used in
        StyleGAN2.

        Args:
            dim (int): Number of input channels
            dim_out (int): Number of output channels
            dim_embed (int): Embedding dimension for adaptive weights
            kernel (int): Convolution kernel size
            demod (bool): Whether to apply weight demodulation (default: True)
            stride (int): Convolution stride (default: 1)
            dilation (int): Convolution dilation (default: 1)
            eps (float): Small constant for numerical stability (default: 1e-8)
            num_conv_kernels (int, optional): Number of adaptive convolution kernels (default: 1)

        Note:
            If num_conv_kernels is greater than 1, the layer is adaptive and the weights are determined
            by a linear transformation of an embedding tensor, followed by a softmax operation to select
            which kernel to use for each input.
        """
        super().__init__()
        self.eps = eps

        self.dim_out = dim_out
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.adaptive = num_conv_kernels > 1

        self.to_mod = nn.Linear(
            dim_embed,
            dim)  # Linear layer to transform embedding to modulating factor
        self.to_adaptive_weight = nn.Linear(
            dim_embed, num_conv_kernels) if self.adaptive else None

        # Initialize weights with normal distribution
        self.weights = nn.Parameter(
            torch.randn((num_conv_kernels, dim_out, dim, kernel, kernel)))
        nn.init.kaiming_normal_(self.weights,
                                a=0,
                                mode='fan_in',
                                nonlinearity='leaky_relu')

        self.demod = demod

    def forward(self, fmap: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """Forward pass of Adaptive Convolutional Layer with Modulated Weights.

        Args:
            fmap (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width)
            embed (torch.Tensor): Embedding tensor with shape (batch_size, embedding_dim)

        Returns:
            (torch.Tensor): Output tensor with shape (batch_size, out_channels, height_out, width_out)
        """
        b, h = fmap.shape[0], fmap.shape[-2]

        weights = self.weights

        if self.adaptive:
            # If using adaptive weights, repeat weights for each sample in the batch
            weights = repeat(weights, '... -> b ...', b=b)

            # Determine an adaptive weight and 'select' the kernel to use with softmax
            selections = self.to_adaptive_weight(embed).softmax(dim=-1)
            selections = rearrange(selections, 'b n -> b n 1 1 1 1')

        # Apply weight modulation
        mod = self.to_mod(embed)
        mod = rearrange(mod, 'b i -> b 1 i 1 1')

        weights = weights * (mod + 1)

        if self.demod:
            inv_norm = reduce(weights**2, 'b o i k1 k2 -> b o 1 1 1',
                              'sum').clamp(min=self.eps).rsqrt()
            weights = weights * inv_norm

        fmap = rearrange(fmap, 'b c h w -> 1 (b c) h w')

        weights = rearrange(weights, 'b o ... -> (b o) ...')

        padding = get_same_padding(h, self.kernel, self.dilation, self.stride)
        fmap = F.conv2d(fmap, weights, padding=padding, groups=b)

        return rearrange(fmap, '1 (b o) ... -> b o ...', b=b)


class Attention(nn.Module):
    """Compute scaled dot product attention given query, key, value and an
    optional mask tensor.

    Parameters:
        dropout (float): Dropout probability applied to the attention weights.

    Inputs:
        query (Tensor): Query tensor of shape (batch_size, query_len, d_model).
        key (Tensor): Key tensor of shape (batch_size, key_len, d_model).
        value (Tensor): Value tensor of shape (batch_size, value_len, d_model).
        mask (Tensor, optional): Mask tensor of shape (batch_size, query_len, key_len), with optional broadcast dimensions.

    Outputs:
        output (Tensor): Tensor of shape (batch_size, query_len, d_model).
        attention_weights (Tensor): Tensor of shape (batch_size, nhead, query_len, key_len).
    """
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        # Compute the scaled dot product between the query and key tensors, scaled by the square root of the feature dimension.
        # The attention scores are obtained by multiplying the dot product by the inverse square root of the dimension.
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(query.size(-1), dtype=query.dtype))
        # If a mask tensor is provided, add a large negative number to the scores corresponding to masked positions.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # Apply the softmax function along the key dimension to obtain the attention weights.
        attention_weights = nn.functional.softmax(scores, dim=-1)
        # Apply dropout to the attention weights.
        attention_weights = self.dropout(attention_weights)
        # Compute the weighted sum of the value tensor using the attention weights.
        output = torch.matmul(attention_weights, value)
        return output, attention_weights


class SelfAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, mask_self_value=-1e2):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads

        self.mask_self_value = mask_self_value

        self.norm = ChannelRMSNorm(dim)
        self.to_qk = nn.Conv2d(dim, dim_inner, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim_inner, 1, bias=False)

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias=False)

    def forward(self, fmap):
        """einstein notation.

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch, device = fmap.shape[0], fmap.device

        fmap = self.norm(fmap)

        x, y = fmap.shape[-2:]

        h = self.heads

        qk, v = self.to_qk(fmap), self.to_v(fmap)
        qk, v = map(
            lambda t: rearrange(
                t, 'b (h d) x y -> (b h) (x y) d', h=self.heads), (qk, v))

        q, k = qk, qk

        # add a null key / value, so network can choose to pay attention to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b=batch),
                     self.null_kv)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # l2 distance

        sim = -torch.cdist(q, k, p=2) * self.scale

        # following what was done in reformer for shared query / key space
        # omit attention to self

        self_mask = torch.eye(sim.shape[-2], device=device, dtype=torch.bool)
        self_mask = F.pad(self_mask, (1, 0), value=False)

        sim = sim.masked_fill(self_mask, self.mask_self_value)

        # attention

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x=x, y=y, h=h)

        return self.to_out(out)


class MultiHeadAttention(nn.Module):
    """Compute multi-head attention given query, key, and value tensors.

    Parameters:
        d_model (int): Input feature dimensionality.
        nhead (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout probability applied to the attention weights. Defaults to 0.1.

    Inputs:
        query (Tensor): Query tensor of shape (batch_size, query_len, d_model).
        key (Tensor): Key tensor of shape (batch_size, key_len, d_model).
        value (Tensor): Value tensor of shape (batch_size, value_len, d_model).
        mask (Tensor, optional): Mask tensor of shape (batch_size, query_len, key_len), with optional broadcast dimensions.

    Outputs:
        output (Tensor): Tensor of shape (batch_size, query_len, d_model).
        attention_weights (Tensor): Tensor of shape (batch_size, nhead, query_len, key_len).
    """
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, 'Input dimensionality must be divisible by the number of attention heads.'
        # We assume d_v always equals d_k
        self.d_k = d_model // nhead
        self.nhead = nhead
        # Define the linear projections
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(dropout=dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """Take in and process masked source/target sequences.

        Args:
            query (tensor): query tensor with shape (batch_size, seq_len, d_model)
            key (tensor): key tensor with shape (batch_size, seq_len, d_model)
            value (tensor): value tensor with shape (batch_size, seq_len, d_model)
            mask (tensor, optional): tensor with shape (batch_size, 1, 1, seq_len) representing the mask
                                     to be applied to the attention scores. Defaults to None.

        Returns:
            tensor: output tensor with shape (batch_size, seq_len, d_model)
            tensor: attention tensor with shape (batch_size, nhead, seq_len, seq_len)
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            linear(x).view(nbatches, -1, self.nhead, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self_attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1,
                                                self.nhead * self.d_k)
        # 4) linear proj output
        output = self.output_linear(x)
        return output, self_attn


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.attn = SelfAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.ff = FeedForward(dim=dim, mult=ff_mult, channel_first=True)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, dim_context, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.attn = CrossAttention(dim=dim,
                                   dim_context=dim_context,
                                   dim_head=dim_head,
                                   heads=heads)
        self.ff = FeedForward(dim=dim, mult=ff_mult, channel_first=True)

    def forward(self, x, context, mask=None):
        x = self.attn(x, context=context, mask=mask) + x
        x = self.ff(x) + x
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads

        self.norm = ChannelRMSNorm(dim)
        self.norm_context = RMSNorm(dim)

        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias=False)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias=False)
        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias=False)

    def forward(self, fmap, context):
        """einstein notation.

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """

        fmap = self.norm(fmap)
        context = self.norm_context(context)

        x, y = fmap.shape[-2:]

        h = self.heads

        q, k, v = (self.to_q(fmap), *self.to_kv(context).chunk(2, dim=-1))

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                   (k, v))

        q = rearrange(q, 'b (h d) x y -> (b h) (x y) d', h=self.heads)

        sim = -torch.cdist(q, k, p=2) * self.scale  # l2 distance

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x=x, y=y, h=h)

        return self.to_out(out)


# classic transformer attention, stick with l2 distance


class TextAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, mask_self_value=-1e2):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads

        self.mask_self_value = mask_self_value

        self.norm = RMSNorm(dim)
        self.to_qk = nn.Linear(dim, dim_inner, bias=False)
        self.to_v = nn.Linear(dim, dim_inner, bias=False)

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, encodings, mask=None):
        """einstein notation.

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch, device = encodings.shape[0], encodings.device

        encodings = self.norm(encodings)

        h = self.heads

        qk, v = self.to_qk(encodings), self.to_v(encodings)
        qk, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads),
            (qk, v))

        q, k = qk, qk

        # add a null key / value, so network can choose to pay attention to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b=batch),
                     self.null_kv)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # l2 distance

        sim = -torch.cdist(q, k, p=2) * self.scale

        # following what was done in reformer for shared query / key space
        # omit attention to self

        self_mask = torch.eye(sim.shape[-2], device=device, dtype=torch.bool)
        self_mask = F.pad(self_mask, (1, 0), value=False)

        sim = sim.masked_fill(self_mask, self.mask_self_value)

        # key padding mask

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = repeat(mask, 'b n -> (b h) 1 n', h=h)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


# feedforward


def FeedForward(dim, mult=4):
    dim_hidden = int(dim * mult)
    return nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim_hidden), nn.GELU(),
                         nn.Linear(dim_hidden, dim))


# transformer


class Transformer(nn.Module):
    def __init__(self, dim, depth, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    TextAttention(dim=dim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=dim, mult=ff_mult)
                ]))

        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x

        return self.norm(x)


# text encoder


@beartype
class TextEncoder(nn.Module):
    def __init__(
        self,
        *,
        clip: OpenClipAdapter,
        dim,
        depth,
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.clip = clip
        self.learned_global_token = nn.Parameter(torch.randn(dim))

        self.project_in = nn.Linear(
            clip.dim_latent, dim) if clip.dim_latent != dim else nn.Identity()

        self.transformer = Transformer(dim=dim,
                                       depth=depth,
                                       dim_head=dim_head,
                                       heads=heads)

    def forward(self, texts: List[str]):
        _, text_encodings = self.clip.embed_texts(texts)
        mask = (text_encodings != 0.).any(dim=-1)

        text_encodings = self.project_in(text_encodings)

        mask_with_global = F.pad(mask, (1, 0), value=True)

        batch = text_encodings.shape[0]
        global_tokens = repeat(self.learned_global_token, 'd -> b d', b=batch)

        text_encodings, ps = pack([global_tokens, text_encodings], 'b * d')

        text_encodings = self.transformer(text_encodings,
                                          mask=mask_with_global)

        global_tokens, text_encodings = unpack(text_encodings, ps, 'b * d')

        return global_tokens, text_encodings, mask


# style mapping network


class StyleNetwork(nn.Module):
    """in the stylegan2 paper, they control the learning rate by multiplying
    the parameters by a constant, but we can use another trick here from
    attention literature."""
    def __init__(self, dim, depth, dim_text_latent=0, frac_gradient=0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend(
                [nn.Linear(dim + dim_text_latent, dim),
                 leaky_relu()])

        self.net = nn.Sequential(*layers)
        self.frac_gradient = frac_gradient
        self.dim_text_latent = dim_text_latent

    def forward(self, x, text_latent=None):
        grad_frac = self.frac_gradient

        if self.dim_text_latent:
            assert exists(text_latent)
            x = torch.cat((x, text_latent), dim=-1)

        x = F.normalize(x, dim=1)
        out = self.net(x)

        return out * grad_frac + (1 - grad_frac) * out.detach()


# gan
@beartype
class Generator(nn.Module):
    def __init__(
            self,
            *,
            dim,
            image_size,
            dim_max=8192,
            capacity=16,
            channels=3,
            style_network: Optional[StyleNetwork] = None,
            text_encoder: Optional[TextEncoder] = None,
            dim_latent=512,
            self_attn_resolutions: Tuple[int] = (32, 16),
            self_attn_dim_head=64,
            self_attn_heads=8,
            self_ff_mult=4,
            cross_attn_resolutions: Tuple[int] = (32, 16),
            cross_attn_dim_head=64,
            cross_attn_heads=8,
            cross_ff_mult=4,
            num_conv_kernels=2,  # the number of adaptive conv kernels
            use_glu=False,
            num_skip_layers_excite=0,
            unconditional=False):
        super().__init__()
        self.dim = dim
        self.channels = channels

        self.style_network = style_network
        self.text_encoder = text_encoder

        self.unconditional = unconditional
        assert not (unconditional and exists(text_encoder))
        assert not (unconditional and exists(style_network)
                    and style_network.dim_text_latent > 0)

        assert is_power_of_two(image_size)
        num_layers = int(log2(image_size) - 1)
        self.num_layers = num_layers

        # generator requires convolutions conditioned by the style vector
        # and also has N convolutional kernels adaptively selected (one of the only novelties of the paper)

        is_adaptive = num_conv_kernels > 1
        dim_kernel_mod = num_conv_kernels if is_adaptive else 0

        style_embed_split_dims = []

        adaptive_conv = partial(AdaptiveConv2DMod,
                                kernel=3,
                                num_conv_kernels=num_conv_kernels)

        # initial 4x4 block and conv

        self.init_block = nn.Parameter(torch.randn(dim_latent, 4, 4))
        self.init_conv = adaptive_conv(dim_latent, dim_latent)

        style_embed_split_dims.extend([dim_latent, dim_kernel_mod])

        # main network

        num_layers = int(log2(image_size) - 1)
        self.num_layers = num_layers

        resolutions = image_size / ((2**torch.arange(num_layers)))
        resolutions = resolutions.long().tolist()

        dim_layers = (2**(torch.arange(num_layers) + 1)) * capacity
        dim_layers.clamp_(max=dim_max)

        dim_layers = torch.flip(dim_layers, (0, ))
        dim_layers = F.pad(dim_layers, (1, 0), value=dim_latent)

        dim_layers = dim_layers.tolist()
        dim_pairs = list(zip(dim_layers[:-1], dim_layers[1:]))

        self.num_skip_layers_excite = num_skip_layers_excite

        self.layers = nn.ModuleList([])

        # go through layers and construct all parameters

        for ind, ((dim_in, dim_out),
                  resolution) in enumerate(zip(dim_pairs, resolutions)):
            is_last = (ind + 1) == len(dim_pairs)
            should_upsample = not is_last
            should_skip_layer_excite = num_skip_layers_excite > 0 and (
                ind + num_skip_layers_excite) < len(dim_pairs)

            has_self_attn = resolution in self_attn_resolutions
            has_cross_attn = resolution in cross_attn_resolutions and not unconditional

            skip_squeeze_excite = None
            if should_skip_layer_excite:
                dim_skip_in, _ = dim_pairs[ind + num_skip_layers_excite]
                skip_squeeze_excite = SqueezeExcite(dim_in, dim_skip_in)

            mult = 2 if use_glu else 1
            act_fn = partial(nn.GLU, dim=1) if use_glu else leaky_relu

            resnet_block = nn.ModuleList([
                adaptive_conv(dim_in, dim_out * mult),
                act_fn(),
                adaptive_conv(dim_out, dim_out * mult),
                act_fn()
            ])

            to_rgb = adaptive_conv(dim_out, channels)

            self_attn = cross_attn = rgb_upsample = upsample = None

            if should_upsample:
                upsample = Upsample(dim_out)
                rgb_upsample = Upsample(channels)

            if has_self_attn:
                self_attn = SelfAttentionBlock(dim_out)

            if has_cross_attn:
                cross_attn = CrossAttentionBlock(dim_out,
                                                 dim_context=text_encoder.dim)

            style_embed_split_dims.extend([
                dim_in,  # for first conv in resnet block
                dim_kernel_mod,  # first conv kernel selection
                dim_out,  # second conv in resnet block
                dim_kernel_mod,  # second conv kernel selection
                dim_out,  # to RGB conv
                dim_kernel_mod,  # RGB conv kernel selection
            ])

            self.layers.append(
                nn.ModuleList([
                    skip_squeeze_excite, resnet_block, to_rgb, self_attn,
                    cross_attn, upsample, rgb_upsample
                ]))

        # determine the projection of the style embedding to convolutional modulation weights (+ adaptive kernel selection weights) for all layers

        self.style_to_conv_modulations = nn.Linear(style_network.dim,
                                                   sum(style_embed_split_dims))
        self.style_embed_split_dims = style_embed_split_dims

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self,
                noise=None,
                styles=None,
                texts: Optional[List[str]] = None,
                global_text_tokens=None,
                fine_text_tokens=None,
                text_mask=None,
                batch_size=1):
        # take care of text encodings
        # which requires global text tokens to adaptively select the kernels from the main contribution in the paper
        # and fine text tokens to attend to using cross attention

        if not self.unconditional:
            if exists(texts):
                assert exists(self.text_encoder)
                global_text_tokens, fine_text_tokens, text_mask = self.text_encoder(
                    texts)
            else:
                assert all([
                    *map(exists,
                         (global_text_tokens, fine_text_tokens, text_mask))
                ])
        else:
            assert not any(
                [*map(exists, (texts, global_text_tokens, fine_text_tokens))])

        # determine styles

        if not exists(styles):
            assert exists(self.style_network)
            noise = default(
                noise, torch.randn((batch_size, self.dim), device=self.device))
            styles = self.style_network(noise, global_text_tokens)

        # project styles to conv modulations

        conv_mods = self.style_to_conv_modulations(styles)
        conv_mods = conv_mods.split(self.style_embed_split_dims, dim=-1)
        conv_mods = iter(conv_mods)

        # prepare initial block

        batch_size = styles.shape[0]

        x = repeat(self.init_block, 'c h w -> b c h w', b=batch_size)
        x = self.init_conv(x, mod=next(conv_mods), kernel_mod=next(conv_mods))

        rgb = torch.zeros((batch_size, self.channels, 4, 4),
                          device=self.device,
                          dtype=x.dtype)

        # skip layer squeeze excitations

        excitations = [None] * self.num_skip_layers_excite

        # main network

        for squeeze_excite, (
                resnet_conv1, act1, resnet_conv2, act2
        ), to_rgb_conv, self_attn, cross_attn, upsample, upsample_rgb in self.layers:

            if exists(squeeze_excite):
                skip_excite = squeeze_excite(x)
                excitations.append(skip_excite)

            excite = safe_unshift(excitations)
            if exists(excite):
                x = x * excite

            x = resnet_conv1(x,
                             mod=next(conv_mods),
                             kernel_mod=next(conv_mods))
            x = act1(x)
            x = resnet_conv2(x,
                             mod=next(conv_mods),
                             kernel_mod=next(conv_mods))
            x = act2(x)

            if exists(self_attn):
                x = self_attn(x)

            if exists(cross_attn):
                x = cross_attn(x, context=fine_text_tokens, mask=text_mask)

            rgb = rgb + to_rgb_conv(
                x, mod=next(conv_mods), kernel_mod=next(conv_mods))

            if exists(upsample):
                x = upsample(x)

            if exists(upsample_rgb):
                rgb = upsample_rgb(rgb)

        return rgb


# discriminator


class Predictor(nn.Module):
    def __init__(self, dim, depth=4, num_conv_kernels=2, unconditional=False):
        super().__init__()
        self.unconditional = unconditional
        self.residual_fn = nn.Conv2d(dim, dim, 1)
        self.layers = nn.ModuleList([])

        klass = nn.Conv2d if unconditional else partial(
            AdaptiveConv2DMod, num_conv_kernels=num_conv_kernels)

        for ind in range(depth):
            self.layers.append(klass(dim, dim, 1))

        self.to_logits = nn.Conv2d(dim, 1, 1)

    def forward(self, x, mod=None, kernel_mod=None):
        residual = self.residual_fn(x)

        for layer in self.layers:
            kwargs = dict()
            if not self.unconditional:
                kwargs = dict(mod=mod, kernel_mod=kernel_mod)

            x = layer(x, **kwargs)

        x = x + residual
        return self.to_logits(x)


@beartype
class Discriminator(nn.Module):
    def __init__(self,
                 *,
                 dim,
                 image_size,
                 capacity=16,
                 dim_max=8192,
                 channels=3,
                 attn_resolutions: Tuple[int] = (32, 16),
                 attn_dim_head=64,
                 attn_heads=8,
                 ff_mult=4,
                 text_encoder: Optional[TextEncoder] = None,
                 text_dim=None,
                 multiscale_input_resolutions: Tuple[int] = (64, 32, 16, 8),
                 multiscale_output_resolutions: Tuple[int] = (32, 16, 8, 4),
                 resize_mode='bilinear',
                 num_conv_kernels=2,
                 use_glu=False,
                 num_skip_layers_excite=0,
                 unconditional=False):
        super().__init__()
        self.unconditional = unconditional
        assert not (unconditional and exists(text_encoder))

        assert is_power_of_two(image_size)
        assert all([*map(is_power_of_two, attn_resolutions)])

        assert all([*map(is_power_of_two, multiscale_input_resolutions)])
        assert all([*map(lambda t: t >= 4, multiscale_input_resolutions)])
        self.multiscale_input_resolutions = multiscale_input_resolutions

        assert all([*map(is_power_of_two, multiscale_output_resolutions)])
        assert all([*map(lambda t: t >= 4, multiscale_output_resolutions)])

        assert max(multiscale_input_resolutions) > max(
            multiscale_output_resolutions)
        assert min(multiscale_input_resolutions) > min(
            multiscale_output_resolutions)

        self.multiscale_output_resolutions = multiscale_output_resolutions

        self.resize_mode = resize_mode

        num_layers = int(log2(image_size) - 1)
        self.num_layers = num_layers

        resolutions = image_size / ((2**torch.arange(num_layers)))
        resolutions = resolutions.long().tolist()

        dim_layers = (2**(torch.arange(num_layers) + 1)) * capacity
        dim_layers = F.pad(dim_layers, (1, 0), value=channels)
        dim_layers.clamp_(max=dim_max)

        dim_layers = dim_layers.tolist()
        dim_last = dim_layers[-1]
        dim_pairs = list(zip(dim_layers[:-1], dim_layers[1:]))

        self.num_skip_layers_excite = num_skip_layers_excite

        self.residual_scale = 2**-0.5
        self.layers = nn.ModuleList([])

        predictor_dims = []
        dim_kernel_attn = (num_conv_kernels if num_conv_kernels > 1 else 0)

        for ind, ((dim_in, dim_out),
                  resolution) in enumerate(zip(dim_pairs, resolutions)):
            is_first = ind == 0
            is_last = (ind + 1) == len(dim_pairs)
            should_downsample = not is_last
            should_skip_layer_excite = not is_first and num_skip_layers_excite > 0 and (
                ind + num_skip_layers_excite) < len(dim_pairs)

            has_attn = resolution in attn_resolutions
            has_multiscale_input = resolution in multiscale_input_resolutions
            has_multiscale_output = resolution in multiscale_output_resolutions

            skip_squeeze_excite = None
            if should_skip_layer_excite:
                dim_skip_in, _ = dim_pairs[ind + num_skip_layers_excite]
                skip_squeeze_excite = SqueezeExcite(dim_in, dim_skip_in)

            dim_in = dim_in + (channels if has_multiscale_input else 0)

            residual_conv = nn.Conv2d(dim_in,
                                      dim_out,
                                      1,
                                      stride=(2 if should_downsample else 1))

            mult = 2 if use_glu else 1
            act_fn = partial(nn.GLU, dim=1) if use_glu else leaky_relu

            resnet_block = nn.Sequential(conv2d_3x3(dim_in, dim_out * mult),
                                         act_fn(),
                                         conv2d_3x3(dim_out, dim_out * mult),
                                         act_fn())

            multiscale_output_predictor = None
            if has_multiscale_output:
                multiscale_output_predictor = Predictor(
                    dim_out,
                    num_conv_kernels=num_conv_kernels,
                    unconditional=unconditional)
                predictor_dims.extend([dim_out, dim_kernel_attn])

            self.layers.append(
                nn.ModuleList([
                    skip_squeeze_excite,
                    resnet_block,
                    residual_conv,
                    SelfAttentionBlock(dim_out,
                                       heads=attn_heads,
                                       dim_head=attn_dim_head,
                                       ff_mult=ff_mult) if has_attn else None,
                    multiscale_output_predictor,
                    Downsample(dim_out) if should_downsample else None,
                ]))

        self.to_logits = nn.Sequential(conv2d_3x3(dim_last, dim_last),
                                       leaky_relu(),
                                       Rearrange('b c h w -> b (c h w)'),
                                       nn.Linear(dim_last * (4**2), 1),
                                       Rearrange('b 1 -> b'))

        # take care of text conditioning in the multiscale predictor branches

        assert unconditional or (exists(text_dim) ^ exists(text_encoder))

        if not unconditional:
            self.text_encoder = text_encoder

            self.text_dim = default(text_dim, text_encoder.dim)

            self.predictor_dims = predictor_dims
            self.text_to_conv_conditioning = nn.Linear(
                self.text_dim, sum(predictor_dims)) if exists(
                    self.text_dim) else None

    def forward(self,
                images,
                texts: Optional[List[str]] = None,
                text_embeds=None):
        if not self.unconditional:
            assert exists(texts) ^ exists(text_embeds)

            if exists(texts):
                assert exists(self.text_encoder)
                text_embeds, *_ = self.text_encoder(texts)

            conv_mods = self.text_to_conv_conditioning(text_embeds).split(
                self.predictor_dims, dim=-1)
            conv_mods = iter(conv_mods)
        else:
            assert not any([*map(exists, (texts, text_embeds))])

        x = images

        # hold multiscale outputs

        multiscale_outputs = []

        # excitations

        excitations = [None] * (
            self.num_skip_layers_excite + 1
        )  # +1 since first image in pixel space is not excited

        for squeeze_excite, block, residual_fn, attn, predictor, downsample in self.layers:
            resolution = x.shape[-1]

            if exists(squeeze_excite):
                skip_excite = squeeze_excite(x)
                excitations.append(skip_excite)

            excite = safe_unshift(excitations)
            if exists(excite):
                x = x * excite

            if resolution in self.multiscale_input_resolutions:
                resized_images = F.interpolate(images,
                                               resolution,
                                               mode=self.resize_mode)
                x = torch.cat((resized_images, x), dim=1)

            residual = residual_fn(x)
            x = block(x)

            if exists(attn):
                x = attn(x)

            if exists(predictor):
                pred_kwargs = dict()
                if not self.unconditional:
                    pred_kwargs = dict(mod=next(conv_mods),
                                       kernel_mod=next(conv_mods))

                multiscale_outputs.append(predictor(x, **pred_kwargs))

            if exists(downsample):
                x = downsample(x)

            x = x + residual
            x = x * self.residual_scale

        logits = self.to_logits(x)

        return logits, multiscale_outputs


# gan
@beartype
class GigaGAN(nn.Module):
    def __init__(self, generator: Generator, discriminator: Discriminator):
        super().__init__()

    def forward(self, x):
        return x
