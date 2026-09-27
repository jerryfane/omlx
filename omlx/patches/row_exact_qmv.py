# SPDX-License-Identifier: Apache-2.0
"""Multi-row quantized projection with one-row decode arithmetic per row.

Stock ``quantized_matmul`` switches kernels with the row count (``qmv_fast``
or ``qmv`` at one row, ``qmv_wide`` and then tiled qmm for more), so a
speculative verify row and the serial decode step for the same token get
different bits from the same projection. Here every row runs MLX 0.32.2's
one-row ``qmv_fast`` (K a multiple of 512) or ``qmv`` traversal, transcribed
in ``moe_verify_gather``, so row ``r`` of the output equals
``quantized_matmul(x[r:r+1])`` bit for bit.

One threadgroup computes an 8-column output tile for every row: each weight
pack is read once per block and applied to all rows, so a verify block costs
about one weight pass. Every (row, column) accumulator still sums its K
blocks in qmv order and finishes with the same ``simd_sum``.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx
import mlx.nn as nn

from .moe_verify_gather import _BITS, _GROUP_SIZES, _HEADER

MAX_ROWS = 8

# Per row this is qmv_fast_impl (FAST) or the full-tile branch of qmv_impl,
# exactly as moe_verify_gather runs them per (row, expert) pair; the row loop
# sits inside each K block so the block's weight bytes are fetched once.
_SOURCE = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;

    const int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K_SIZE / GS;
    const int out_row = int(threadgroup_position_in_grid.z) * BN +
        int(simd_gid) * RESULTS_PER_SIMDGROUP;

    const device uint8_t* ws = (const device uint8_t*)w +
        out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc = scales + out_row * in_vec_size_g +
        int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs = biases + out_row * in_vec_size_g +
        int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xp = x + int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[ROWS][RESULTS_PER_SIMDGROUP];
    for (int i = 0; i < ROWS; i++) {
      for (int r = 0; r < RESULTS_PER_SIMDGROUP; r++) {
        result[i][r] = 0;
      }
    }

    int k = 0;
    const int full_limit = FAST ? K_SIZE : K_SIZE - BLOCK_SIZE;
    for (; k < full_limit; k += BLOCK_SIZE) {
      for (int i = 0; i < ROWS; i++) {
        float sum = load_vector<T>(xp + i * K_SIZE, x_thread);
        for (int r = 0; r < RESULTS_PER_SIMDGROUP; r++) {
          const device uint8_t* wl = ws + r * in_vec_size_w;
          float s = sc[r * in_vec_size_g];
          float b = bs[r * in_vec_size_g];
          result[i][r] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
        }
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xp += BLOCK_SIZE;
    }
    if (!FAST) {
      const int remaining = clamp(
          int(K_SIZE - k - int(simd_lid) * VALUES_PER_THREAD),
          0,
          VALUES_PER_THREAD);
      if (remaining > 0) {
        for (int i = 0; i < ROWS; i++) {
          float sum = load_vector_safe<T>(xp + i * K_SIZE, x_thread, remaining);
          for (int r = 0; r < RESULTS_PER_SIMDGROUP; r++) {
            const device uint8_t* wl = ws + r * in_vec_size_w;
            float s = sc[r * in_vec_size_g];
            float b = bs[r * in_vec_size_g];
            result[i][r] += qdot_n(wl, x_thread, s, b, sum, remaining);
          }
        }
      }
    }

    for (int i = 0; i < ROWS; i++) {
      for (int r = 0; r < RESULTS_PER_SIMDGROUP; r++) {
        float v = simd_sum(result[i][r]);
        if (simd_lid == 0) {
          y[i * N_SIZE + out_row + r] = static_cast<T>(v);
        }
      }
    }
"""


@cache
def _kernel(bits: int, group_size: int, fast: bool):
    header = (
        _HEADER.replace("__BITS__", str(bits))
        .replace("__GS__", str(group_size))
        .replace("__FAST__", "1" if fast else "0")
    )
    return mx.fast.metal_kernel(
        name=f"omlx_row_exact_qmv_b{bits}_gs{group_size}_{int(fast)}",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=header,
        source=_SOURCE,
    )


def _kernel_supported(linear: nn.QuantizedLinear, x: mx.array) -> bool:
    bits, group_size = linear.bits, linear.group_size
    if (
        getattr(linear, "mode", "affine") != "affine"
        or bits not in _BITS
        or group_size not in _GROUP_SIZES
        or linear.biases is None
        or x.dtype not in (mx.bfloat16, mx.float16)
        or linear.scales.dtype != x.dtype
        or linear.biases.dtype != x.dtype
    ):
        return False
    n = int(linear.weight.shape[0])
    k = int(linear.scales.shape[-1]) * group_size
    values_per_thread = (8 if bits == 5 else (4 if bits == 6 else 32 // bits)) * (
        2 if k % 512 == 0 else 1
    )
    return (
        x.shape[-1] == k
        and n % 8 == 0
        and k % 8 == 0
        and group_size % values_per_thread == 0
    )


def quantized_linear(linear: nn.QuantizedLinear, x: mx.array) -> mx.array:
    """``linear(x)`` for ``x`` of shape [..., K], each row with one-row arithmetic.

    Shapes the kernel does not cover (N not a multiple of 8, non-affine
    modes, more than MAX_ROWS rows) run one stock one-row call per row,
    which is the serial decode call itself.
    """
    lead = x.shape[:-1]
    rows = 1
    for size in lead:
        rows *= int(size)
    if rows <= 1:
        return linear(x)
    k = int(x.shape[-1])
    flat = mx.contiguous(x.reshape(rows, k))
    if rows <= MAX_ROWS and _kernel_supported(linear, x):
        n = int(linear.weight.shape[0])
        y = _kernel(int(linear.bits), int(linear.group_size), k % 512 == 0)(
            inputs=[flat, linear.weight, linear.scales, linear.biases],
            template=[("T", x.dtype), ("K_SIZE", k), ("N_SIZE", n), ("ROWS", rows)],
            grid=(32, 2, n // 8),
            threadgroup=(32, 2, 1),
            output_shapes=[(rows, n)],
            output_dtypes=[x.dtype],
        )[0]
        if "bias" in linear:
            y = y + linear["bias"]
    else:
        y = mx.concatenate([linear(flat[r : r + 1][None])[0] for r in range(rows)], axis=0)
    return y.reshape(*lead, -1)


__all__ = ["MAX_ROWS", "quantized_linear"]
