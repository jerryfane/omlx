# SPDX-License-Identifier: Apache-2.0
"""Fused routed experts for one-token Qwen3.5-MoE-family decode.

After the fused router, a one-token MoE block runs its routed experts as the
gate+up ``gather_qmm``, the compiled SwiGLU and the down ``gather_qmm`` (each
``gather_qmm`` behind an ``arange`` for its row indices), then one combine
launch for the score-weighted sum plus the gated shared expert. At batch-one
decode these launches are short and depend on each other, so this patch runs
the same arithmetic in two launches:

1. gate+up with a SwiGLU epilogue. Each simdgroup computes the gate rows and
   the matching up rows of one expert with MLX's ``qmv_fast`` lane partition
   and add order, rounds both to the activation dtype, then applies MLX's
   ``Sigmoid`` and the two multiplies of the compiled ``swiglu`` in its order.
2. down with the combine. Simdgroup ``j`` of a threadgroup runs the stock
   ``qmv`` work (including its guarded K tail) of selected expert ``j`` for
   the threadgroup's rows. Each row is rounded, then the combine of
   ``qwen35_moe_router.fused_moe_combine`` follows: score products summed in
   MLX's ``col_reduce_small`` order, plus ``sigmoid(shared_gate) * shared``.

The result is bit-identical to the composed path. The quantized dot products
reuse the MLX 0.32.2 transcription in ``moe_verify_gather`` (4, 5, 6 and
8 bits; group size 32, 64 or 128). MLX picks ``qmv_fast`` when K % 512 == 0
and N % 8 == 0 and ``qmv`` otherwise, so only shapes where gate+up takes
``qmv_fast`` and down takes ``qmv`` are routed: one bf16 token, top-k 10,
affine experts with bf16 scales, hidden % 512 == 0 and intermediate
% 512 != 0 (Qwen3.8-Flash-Next: 2560 and 640, 4-bit or oQ5e's 5-bit).
Prefill, verify rows and every other shape keep the original body. If the
first launch fails, the patch disables itself and the block keeps its
composed body. ``OMLX_QWEN35_MOE_ROUTED_DECODE=0`` keeps the composed body.

MLX commits a command buffer once the inputs bound to it exceed its size cap
(50 MB by default), counting each input array whole. The stacked expert
weights are hundreds of MB, so every launch that binds them ends a command
buffer (~10-20 us of host CPU and a GPU gap per commit). The kernels
therefore bind a one-expert view that shares the stacked array's buffer at
offset 0 and index the other experts from it. The weights are resident model
parameters, so the cap has nothing to bound here. Only scheduling changes;
``OMLX_QWEN35_MOE_ROUTED_DECODE_VIEWS=0`` binds the whole arrays.
"""

from __future__ import annotations

import logging
import os
from functools import cache
from typing import NamedTuple

import mlx.core as mx
import numpy as np

from .module_cache import cached_per_module
from .moe_verify_gather import _BITS, _GROUP_SIZES
from .moe_verify_gather import _HEADER as _QMV_HEADER

logger = logging.getLogger(__name__)

TOP_K = 10
_GATE_UP_ROWS = 2  # gate rows (and as many up rows) per simdgroup
_GATE_UP_SIMDGROUPS = 2
_DOWN_ROWS = 4
_ENABLED = os.environ.get("OMLX_QWEN35_MOE_ROUTED_DECODE", "1") != "0"
_VIEWS_ENABLED = os.environ.get("OMLX_QWEN35_MOE_ROUTED_DECODE_VIEWS", "1") != "0"
_DISABLED = False
_PROVEN = False

_SIGMOID = r"""
// MLX 0.32.2 Sigmoid, evaluated in T as the compiled swiglu does.
template <typename U>
inline U omlx_mlx_sigmoid(U x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

# One threadgroup per (expert slot z, block of NSG * RPS output rows). Output
# is [TOP_K, N / 2] = silu(gate) * up.
_GATE_UP_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    const size_t expert = size_t(rhs[tid.z]);

    const device uint8_t* ws = (const device uint8_t*)w +
        expert * N * in_vec_size_w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc = scales + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs = biases + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xp = x + int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[2 * RPS] = {0};

    for (int k = 0; k < K; k += BLOCK_SIZE) {
      float sum = load_vector<T>(xp, x_thread);
      for (int row = 0; row < 2 * RPS; row++) {
        const int r = row < RPS ? row : N / 2 + row - RPS;
        const device uint8_t* wl = ws + r * in_vec_size_w;
        float s = sc[r * in_vec_size_g];
        float b = bs[r * in_vec_size_g];
        result[row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xp += BLOCK_SIZE;
    }

    for (int row = 0; row < 2 * RPS; row++) {
      result[row] = simd_sum(result[row]);
    }
    if (simd_lid == 0) {
      device T* yp = y + size_t(tid.z) * (N / 2) + out_row;
      for (int row = 0; row < RPS; row++) {
        T g = static_cast<T>(result[row]);
        T u = static_cast<T>(result[row + RPS]);
        T t = g * omlx_mlx_sigmoid<T>(g);
        yp[row] = t * u;
      }
    }
"""

# Threadgroup (32, TOP_K): simdgroup j computes RPS rows of selected expert j,
# then simdgroup 0 combines the rows with the shared expert. Output is [N].
_DOWN_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row = int(tid.y) * RPS;
    const int slot = int(simd_gid);
    const size_t expert = size_t(rhs[slot]);
    threadgroup T part[10 * RPS];

    const device uint8_t* ws = (const device uint8_t*)w +
        expert * N * in_vec_size_w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc = scales + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs = biases + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xp = x + slot * K + int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[RPS] = {0};

    int k = 0;
    for (; k < K - BLOCK_SIZE; k += BLOCK_SIZE) {
      float sum = load_vector<T>(xp, x_thread);
      for (int row = 0; row < RPS; row++) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float s = sc[row * in_vec_size_g];
        float b = bs[row * in_vec_size_g];
        result[row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xp += BLOCK_SIZE;
    }
    const int remaining = clamp(
        int(K - k - int(simd_lid) * VALUES_PER_THREAD), 0, VALUES_PER_THREAD);
    if (remaining > 0) {
      float sum = load_vector_safe<T>(xp, x_thread, remaining);
      for (int row = 0; row < RPS; row++) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float s = sc[row * in_vec_size_g];
        float b = bs[row * in_vec_size_g];
        result[row] += qdot_n(wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0; row < RPS; row++) {
      result[row] = simd_sum(result[row]);
    }
    if (simd_lid == 0) {
      for (int row = 0; row < RPS; row++) {
        part[slot * RPS + row] = static_cast<T>(result[row]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0 && int(simd_lid) < RPS) {
      // fused_moe_combine's arithmetic: each product and each add rounds to
      // T; the k-sum is MLX's one-row col_reduce_small (lane j % 8 folds
      // rows j, j + 8 onto +0, then lanes 1..7 add onto lane 0 in order).
      const int h = out_row + int(simd_lid);
      T lane[8];
      for (int l = 0; l < 8; ++l) {
        lane[l] = T(0.0f);
      }
      for (int j = 0; j < 10; ++j) {
        const T p = T(float(part[j * RPS + int(simd_lid)]) * float(scores[j]));
        lane[j % 8] = T(float(p) + float(lane[j % 8]));
      }
      T acc = lane[0];
      for (int l = 1; l < 8; ++l) {
        acc = T(float(lane[l]) + float(acc));
      }
      const float g = float(gate[0]);
      const T e = T(1.0f + float(T(metal::precise::exp(metal::abs(g)))));
      const T sy = T(metal::precise::divide(1.0f, float(e)));
      const T sg = g < 0.0f ? sy : T(1.0f - float(sy));
      const T sh = T(float(sg) * float(shared[h]));
      y[h] = T(float(acc) + float(sh));
    }
"""


def _header(bits: int, group_size: int, fast: bool) -> str:
    return (
        _QMV_HEADER.replace("__BITS__", str(bits))
        .replace("__GS__", str(group_size))
        .replace("__FAST__", "1" if fast else "0")
        + _SIGMOID
    )


@cache
def _gate_up_kernel(bits: int, group_size: int):
    return mx.fast.metal_kernel(
        name=f"omlx_qwen35_moe_gate_up_swiglu_decode_b{bits}_gs{group_size}",
        input_names=["x", "w", "scales", "biases", "rhs"],
        output_names=["y"],
        header=_header(bits, group_size, fast=True),
        source=_GATE_UP_SOURCE,
    )


@cache
def _down_kernel(bits: int, group_size: int):
    return mx.fast.metal_kernel(
        name=f"omlx_qwen35_moe_down_combine_decode_b{bits}_gs{group_size}",
        input_names=["shared", "gate", "x", "w", "scales", "biases", "rhs", "scores"],
        output_names=["y"],
        header=_header(bits, group_size, fast=False),
        source=_DOWN_SOURCE,
    )


def _quantized_ok(layer) -> bool:
    # The type the served decode plan (qwen35_moe_gate_up) runs as a bare
    # gather_qmm; subclasses may override the call.
    from mlx_vlm.models.switch_layers import QuantizedSwitchLinear

    return (
        type(layer) is QuantizedSwitchLinear
        and layer.bits in _BITS
        and layer.group_size in _GROUP_SIZES
        and layer.mode == "affine"
        and "biases" in layer
        and "bias" not in layer
        and layer["scales"].dtype == mx.bfloat16
        and layer["biases"].dtype == mx.bfloat16
    )


def _address(a: mx.array) -> int:
    return np.frombuffer(memoryview(a), dtype=np.uint8).ctypes.data


def _expert_view(a: mx.array) -> mx.array:
    """``a[:1]`` when it shares ``a``'s buffer at offset 0, else ``a``.

    The kernels index every expert from the bound pointer, so a view that
    were copied (or offset) would read the wrong bytes; keep the whole array
    unless the first-expert slice provably aliases it."""
    view = a[:1]
    mx.eval(view)
    whole, first = memoryview(a), memoryview(view)
    if whole.c_contiguous and first.c_contiguous and _address(view) == _address(a):
        return view
    return a


class _Plan(NamedTuple):
    hidden: int
    gate_up_kernel: object
    gate_up_operands: tuple
    gate_up_template: list
    gate_up_grid: tuple
    down_kernel: object
    down_operands: tuple
    down_template: list
    down_grid: tuple
    h_shape: tuple


def _build_plan(switch_mlp) -> _Plan | None:
    """Kernels and operands for one SwitchGLU, or None outside the layout."""
    from mlx_vlm.models.switch_layers import SwiGLU

    if switch_mlp.training or type(switch_mlp.get("activation")) is not SwiGLU:
        return None
    gate_up = switch_mlp.get("gate_up_proj")
    down = switch_mlp.get("down_proj")
    if gate_up is None or down is None:
        return None
    if not (_quantized_ok(gate_up) and _quantized_ok(down)):
        return None
    gu_bits, d_bits = gate_up.bits, down.bits
    hidden = down["weight"].shape[1]
    inter = down["weight"].shape[-1] * 32 // d_bits
    if not (
        hidden % 512 == 0
        and inter % 512 != 0
        and inter % down.group_size == 0
        and gate_up["weight"].shape[1:] == (2 * inter, hidden * gu_bits // 32)
        and down["weight"].shape[1:] == (hidden, inter * d_bits // 32)
        and gate_up["weight"].shape[0] == down["weight"].shape[0]
    ):
        return None
    view = _expert_view if _VIEWS_ENABLED else (lambda a: a)
    rows = _GATE_UP_ROWS * _GATE_UP_SIMDGROUPS
    return _Plan(
        hidden=hidden,
        gate_up_kernel=_gate_up_kernel(gu_bits, gate_up.group_size),
        gate_up_operands=tuple(view(gate_up[k]) for k in ("weight", "scales", "biases")),
        gate_up_template=[
            ("T", mx.bfloat16),
            ("K", hidden),
            ("N", 2 * inter),
            ("RPS", _GATE_UP_ROWS),
            ("NSG", _GATE_UP_SIMDGROUPS),
        ],
        gate_up_grid=(32, _GATE_UP_SIMDGROUPS * inter // rows, TOP_K),
        down_kernel=_down_kernel(d_bits, down.group_size),
        down_operands=tuple(view(down[k]) for k in ("weight", "scales", "biases")),
        down_template=[
            ("T", mx.bfloat16),
            ("K", inter),
            ("N", hidden),
            ("RPS", _DOWN_ROWS),
        ],
        down_grid=(32, TOP_K * hidden // _DOWN_ROWS, 1),
        h_shape=(TOP_K, inter),
    )


def routed_decode_plan(block, x) -> _Plan | None:
    """The block's cached plan when ``x`` is one bf16 token it can route."""
    if _DISABLED or x.dtype != mx.bfloat16 or block.top_k != TOP_K:
        return None
    hidden = x.shape[-1]
    if x.size != hidden:
        return None
    switch_mlp = block.get("switch_mlp")
    if switch_mlp is None:
        return None
    plan = cached_per_module(switch_mlp, "_omlx_routed_decode_plan", _build_plan)
    if plan is None or plan.hidden != hidden:
        return None
    return plan


def routed_decode(plan: _Plan, x, indices, scores, shared, gate):
    """``(switch_mlp(x, indices) * scores[..., None]).sum(axis=-2)
    + mx.sigmoid(gate) * shared`` for one token, in two launches."""
    h = plan.gate_up_kernel(
        inputs=[x, *plan.gate_up_operands, indices],
        template=plan.gate_up_template,
        grid=plan.gate_up_grid,
        threadgroup=(32, _GATE_UP_SIMDGROUPS, 1),
        output_shapes=[plan.h_shape],
        output_dtypes=[mx.bfloat16],
    )[0]
    # The shared expert comes first in the inputs, so its launches are
    # encoded (and can run) before the router's.
    return plan.down_kernel(
        inputs=[shared, gate, h, *plan.down_operands, indices, scores],
        template=plan.down_template,
        grid=plan.down_grid,
        threadgroup=(32, TOP_K, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def apply_qwen35_moe_routed_decode_patch() -> bool:
    """Wrap the router-fused mlx-vlm ``Qwen3_5MoeSparseMoeBlock`` call.

    Needs ``qwen35_moe_router`` applied first: the fast arm reuses its fused
    routing launch, so it selects the same experts with the same scores as
    the body it replaces."""
    if not _ENABLED or not mx.metal.is_available():
        return False
    try:
        from mlx_vlm.models.qwen3_5_moe import language as vlm_moe
    except ImportError:
        return False
    from .qwen35_moe_router import fused_router_topk, router_eligible

    cls = getattr(vlm_moe, "Qwen3_5MoeSparseMoeBlock", None)
    if cls is None or not getattr(cls, "_omlx_router_fused", False):
        return False
    if getattr(cls, "_omlx_routed_decode", False):
        return True
    orig_call = cls.__call__

    def patched_call(self, x):
        global _DISABLED, _PROVEN
        plan = routed_decode_plan(self, x)
        if plan is None or not router_eligible(x, self.num_experts):
            return orig_call(self, x)
        # Children by item: this runs for every MoE layer of every decode step.
        shared = self["shared_expert"](x)
        shared_gate = self["shared_expert_gate"](x)
        gates = mx.softmax(self["gate"](x), axis=-1, precise=True)
        inds, scores = fused_router_topk(gates, self.top_k)
        if scores.dtype != x.dtype or shared.dtype != x.dtype or shared_gate.dtype != x.dtype:
            return orig_call(self, x)
        try:
            y = routed_decode(plan, x, inds, scores, shared, shared_gate)
            if not _PROVEN:
                # Surface a kernel build failure while the call can still
                # fall back, once per process.
                mx.eval(y)
                _PROVEN = True
                logger.info("Qwen MoE fused routed-expert decode engaged")
        except Exception:
            _DISABLED = True
            logger.warning(
                "fused routed-expert decode failed; composed fallback",
                exc_info=True,
            )
            return orig_call(self, x)
        return y

    patched_call._omlx_routed_decode_original = orig_call
    cls.__call__ = patched_call
    cls._omlx_routed_decode = True
    logger.info("Qwen MoE fused routed-expert decode patch applied")
    return True
