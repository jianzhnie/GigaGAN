import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import pack, rearrange, reduce, repeat, unpack


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
    """
    Channel-wise Root Mean Squared (RMS) Normalization module. 

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
        """
        Forward pass of the ChannelRMSNorm module.

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
    """
    Root Mean Squared (RMS) Normalization module. 

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
        """
        Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, ..., dim).

        Returns:
            torch.Tensor: The normalized and scaled tensor of shape (batch_size, ..., dim).
        """
        # Normalize the input tensor along the last dimension
        normed = F.normalize(x, dim=-1)
        # Scale the normalized tensor by the scaling factor and the learnable scaling parameter
        return normed * self.scale * self.gamma


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
        """Adaptive Convolutional Layer with Modulated Weights, as used in StyleGAN2.

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


# gan
class GigaGAN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
