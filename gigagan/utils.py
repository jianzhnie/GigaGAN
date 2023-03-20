import torch.nn as nn

# helpers


def exists(val):
    return val is not None


# activation functions


def leaky_relu(neg_slope=0.1):
    return nn.LeakyReLU(neg_slope)


# rmsnorm (newer papers show mean-centering in layernorm not necessary)
