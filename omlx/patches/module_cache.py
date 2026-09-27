# SPDX-License-Identifier: Apache-2.0
"""Per-module caches of static decode decisions.

One-token decode re-asks the same layout questions of every layer on every
step (quantization signatures, shapes, dtypes of weights that never change).
``cached_per_module`` answers them once per module and keeps the answer in
the module's ``__dict__``. The entry is keyed on the identity of every value
of the module and of its child modules down to ``depth`` levels (submodules,
weights, scales, biases), the module's training flag and caller ``flags``:
reassigning any of them rebuilds the entry on the next call. The entry keeps
the keyed objects alive, so their ids cannot be reused while it is cached.
Plain configuration attributes of children (bits, group_size, eps, ...) are
construction-time constants and are not re-checked.

Built values must not reference ``module`` itself (only its children and
tensors), so the entry creates no reference cycle through the module.
"""

from __future__ import annotations

from itertools import chain
from typing import Any, Callable

import mlx.nn as nn


def _children(module: nn.Module, depth: int) -> list:
    children: list = []
    level = [module]
    for _ in range(depth):
        level = [
            value
            for parent in level
            for value in dict.values(parent)
            if isinstance(value, nn.Module)
        ]
        children.extend(level)
    return children


def cached_per_module(
    module: nn.Module,
    slot: str,
    build: Callable[[nn.Module], Any],
    *,
    depth: int = 1,
    flags: tuple = (),
) -> Any:
    """``build(module)``, cached in ``module.__dict__[slot]`` (see module docstring)."""
    entry = module.__dict__.get(slot)
    if entry is not None:
        children, _refs, key, value = entry
        if key == (
            module._training,
            flags,
            *map(id, chain(dict.values(module), *map(dict.values, children))),
        ):
            return value
    value = build(module)
    children = _children(module, depth)
    refs = list(chain(dict.values(module), *map(dict.values, children)))
    key = (module._training, flags, *map(id, refs))
    module.__dict__[slot] = (children, refs, key, value)
    return value


__all__ = ["cached_per_module"]
