from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import List, Optional
from einops import rearrange
from torch.autograd import grad as torch_grad

from gigagan.open_clip import OpenClipAdapter


def exists(val):
    return val is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None


def is_power_of_two(n):
    return log2(n).is_integer()


def safe_unshift(arr):
    if len(arr) == 0:
        return None
    return arr.pop(0)


# activation functions


def leaky_relu(neg_slope=0.1):
    return nn.LeakyReLU(neg_slope)


def conv2d_3x3(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 3, padding=1)


# tensor helpers


def log(t, eps=1e-20):
    return t.clamp(min=eps).log()


def gradient_penalty(images, output, weight=10):
    gradients, *_ = torch_grad(outputs=output,
                               inputs=images,
                               grad_outputs=torch.ones_like(output),
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True)

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim=1) - 1)**2).mean()


# hinge gan losses


def gen_hinge_loss(fake):
    return fake.mean()


def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


# auxiliary losses


def aux_matching_loss(real, fake):
    return log(1 + real.exp()) + log(1 + fake.exp())


@beartype
def aux_clip_loss(clip: OpenClipAdapter,
                  images: torch.Tensor,
                  texts: Optional[List[str]] = None,
                  text_embeds: Optional[torch.Tensor] = None):
    assert exists(texts) ^ exists(text_embeds)

    if exists(texts):
        text_embeds = clip.embed_texts(texts)

    return clip.contrastive_loss(images=images, text_embeds=text_embeds)
