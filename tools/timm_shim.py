"""The three helpers `tools/hat_arch.py` imports from `basicsr` and `timm`.

HAT's published `hat_arch.py` imports `to_2tuple` and `trunc_normal_` from
`basicsr.archs.arch_util` and `ARCH_REGISTRY` from `basicsr.utils.registry`.
BasicSR is a training framework and pulling it in to run an eval-mode forward
would drag in a large dependency tree; the two functions it actually uses are
five lines each, and the registry decorator is a no-op here. `trunc_normal_` and
`DropPath` are timm's implementations
(https://github.com/huggingface/pytorch-image-models, Apache-2.0), the same ones
`tools/timm_shim.py` in the sibling swin2sr-rs engine supplies.

`ARCH_REGISTRY.register()` is installed as a decorator that returns its class
unchanged, so `@ARCH_REGISTRY.register()` above `class HAT` works without a
registry to register into - this repository never looks a model up by name.
"""
import types

import torch
import torch.nn as nn


def to_2tuple(x):
    return (x, x) if not isinstance(x, (tuple, list)) else tuple(x)


class DropPath(nn.Module):
    """Per-sample stochastic depth. A no-op at eval, which is the only mode this
    repository runs the network in."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        tensor.normal_(mean, std).clamp_(a * std + mean, b * std + mean)
    return tensor


class _Registry:
    """`basicsr.utils.registry.ARCH_REGISTRY`, as far as this repo needs it."""

    def register(self, *args, **kwargs):
        def deco(cls):
            return cls
        return deco


ARCH_REGISTRY = _Registry()


def install():
    """Register the shim modules under the names `hat_arch` imports.

    Called before `hat_arch` is imported: the published file's imports are part
    of the file, and rewriting them would mean this repository no longer carries
    upstream's source byte for byte.
    """
    import sys

    basicsr = types.ModuleType("basicsr")
    utils = types.ModuleType("basicsr.utils")
    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = ARCH_REGISTRY
    archs = types.ModuleType("basicsr.archs")
    arch_util = types.ModuleType("basicsr.archs.arch_util")
    arch_util.to_2tuple = to_2tuple
    arch_util.trunc_normal_ = trunc_normal_
    utils.registry = registry
    basicsr.utils = utils
    basicsr.archs = archs
    sys.modules.update({
        "basicsr": basicsr,
        "basicsr.utils": utils,
        "basicsr.utils.registry": registry,
        "basicsr.archs": archs,
        "basicsr.archs.arch_util": arch_util,
    })
