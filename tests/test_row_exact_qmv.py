# SPDX-License-Identifier: Apache-2.0
"""Row-exact verify projections: every row equals a one-row decode call.

Lightning MTP greedy output matches MTP-off output only if each verify row's
quantized projection has the serial step's bits; stock multi-row kernels
(qmv_wide, qmm) do not.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import qwen35_verify_qmm, row_exact_qmv

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True)
def disarm():
    yield
    qwen35_verify_qmm.set_verify_qmm_armed(False)


def _linear(k, n, bits, group_size, seed):
    mx.random.seed(seed)
    linear = nn.QuantizedLinear(k, n, bias=False, group_size=group_size, bits=bits)
    weight = (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16)
    linear.weight, linear.scales, linear.biases = mx.quantize(
        weight, group_size=group_size, bits=bits
    )
    return linear


def _serial(linear, x):
    return mx.concatenate([linear(x[:, r : r + 1]) for r in range(x.shape[1])], axis=1)


def _bit_equal(a, b):
    return mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


# (K, N, bits, group_size): Qwen4 oQ5e projections (qmv_fast), the non-fast
# qmv arm (K % 512), partial output tiles (N % 8) and N < 8, other widths.
SHAPES = [
    (2560, 10240, 6, 64),
    (6144, 2560, 5, 128),
    (2560, 248320, 8, 64),
    (640, 2560, 8, 128),
    (2560, 1, 8, 64),
    (2560, 12, 6, 64),
    (320, 1024, 6, 64),
    (2560, 1024, 4, 64),
    (2560, 1024, 5, 32),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rows", [2, 3, 4, 8])
def test_rows_equal_one_row_quantized_matmul(shape, rows):
    k, n, bits, group_size = shape
    linear = _linear(k, n, bits, group_size, seed=k + n + rows)
    x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
    assert _bit_equal(row_exact_qmv.quantized_linear(linear, x), _serial(linear, x))


@pytest.mark.parametrize(
    "k, sizes, bits, group_size",
    [
        (2560, [10240, 6144, 48, 48], 6, 64),  # DeltaNet in_proj qkv/z/b/a
        (2560, [12288, 512, 512], 6, 64),  # attention q/k/v
        (2560, [640, 640], 8, 128),  # shared expert gate/up
        (640, [2560, 1, 12], 8, 128),  # non-fast K with partial tiles
    ],
)
@pytest.mark.parametrize("rows", [2, 4])
def test_grouped_rows_equal_one_row_quantized_matmul(k, sizes, bits, group_size, rows):
    linears = [_linear(k, n, bits, group_size, seed=i + k) for i, n in enumerate(sizes)]
    x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
    outputs = row_exact_qmv.quantized_linears(linears, x)
    assert len(outputs) == len(linears)
    for linear, output in zip(linears, outputs):
        assert _bit_equal(output, _serial(linear, x))


def test_row_exact_arming_routes_multi_row_quantized_linear():
    qwen35_verify_qmm.apply_verify_qmm_patch()
    linear = _linear(2560, 6144, 6, 64, seed=3)
    x = mx.random.normal((1, 4, 2560)).astype(mx.bfloat16)
    reference = _serial(linear, x)
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
    routed = linear(x)
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    assert _bit_equal(routed, reference)
