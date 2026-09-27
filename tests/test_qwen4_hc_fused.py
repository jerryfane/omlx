# SPDX-License-Identifier: Apache-2.0
"""Fused hyper-connection kernels: parity with the canonical path, eligibility, kill switch."""

from __future__ import annotations

import importlib
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


HC, HIDDEN, LOWRANK = 4, 2560, 320
WIDTH = HC * HIDDEN


def _module(bits: int, use_combine: bool = True, hidden: int = HIDDEN):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual, Qwen4ExpRMSNorm

    width = HC * hidden
    module = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
    nn.Module.__init__(module)
    module.hc_count, module.hidden_size, module.hc_lowrank = HC, hidden, LOWRANK
    module.hc_norm = Qwen4ExpRMSNorm(width, group_size=hidden, eps=1e-6)
    module.hc_norm.weight = (mx.random.normal((width,)) * 0.05).astype(mx.bfloat16)
    module.input_mix_weight_down = nn.QuantizedLinear(
        width, LOWRANK, bias=False, group_size=64, bits=bits
    )
    module.input_mix_weight_up = nn.QuantizedLinear(
        LOWRANK, width, bias=False, group_size=64, bits=bits
    )
    if use_combine:
        module.block_inject_weight = nn.QuantizedLinear(
            width, HC, bias=False, group_size=64, bits=bits
        )
    for name in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        projection = getattr(module, name, None)
        if projection is not None:
            # Checkpoint-like statistics: positive scales, small biases. Random-sign scales drive the
            # up-projection gate into saturation where any rounding difference flips whole elements.
            projection.scales = (
                mx.abs(mx.random.normal(projection.scales.shape)) * 0.01 + 0.002
            ).astype(mx.bfloat16)
            projection.biases = (
                mx.random.normal(projection.biases.shape) * 0.005
            ).astype(mx.bfloat16)
    mx.eval(module.parameters())
    return module


def _reference_fp32(module, x):
    def dequant(q):
        return mx.dequantize(
            q.weight, q.scales, q.biases, group_size=q.group_size, bits=q.bits
        ).astype(mx.float32)

    normed = module.hc_norm(x).astype(mx.float32)
    mix = nn.silu((normed @ dequant(module.input_mix_weight_down).T) / HC)
    gate = mx.sigmoid(mix @ dequant(module.input_mix_weight_up).T)
    hidden = module.hidden_size
    mixed = mx.mean(
        gate.reshape(*gate.shape[:-1], HC, hidden)
        * normed.reshape(*normed.shape[:-1], HC, hidden),
        axis=-2,
    )
    if "block_inject_weight" not in module:
        return mixed, None
    return mixed, 2 * mx.sigmoid((normed @ dequant(module.block_inject_weight).T) / HC)


def _ulps(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    ulp = float(mx.abs(b).max().item()) * 2.0**-7
    diff = mx.abs(a - b) / ulp
    return float(diff.max().item()), float(diff.mean().item())


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("rows", [1, 4, 16])
@pytest.mark.parametrize("use_combine", [True, False])
def test_fused_matches_canonical_path(bits, rows, use_combine):
    mx.random.seed(20260905 + bits * 100 + rows)
    _assert_fused_matches_canonical(_module(bits, use_combine), rows, use_combine)


# Sizes the checkpoint never has, chosen so every kernel sees a partial final block:
#   768  -> down tail 256 for 4/5-bit;                          norm and inject loops exact
#   1152 -> down tail 128 (all bits), inject tail 128 (4/5-bit), norm tail 128
#   1344 -> down tail 320 (4/5) / 64 (6/8), inject tail 64,       norm tail 64
#   512, 1536 -> odd multiples of 512, no tails anywhere
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("hidden", [512, 768, 1152, 1344, 1536])
def test_fused_matches_canonical_path_at_other_hidden_sizes(hidden, bits):
    mx.random.seed(20260906 + hidden + bits)
    _assert_fused_matches_canonical(_module(bits, True, hidden=hidden), 16, True)


def _assert_fused_matches_canonical(module, rows, use_combine):
    from mlx_vlm.models.qwen4_exp import hc_fused

    hidden = module.hidden_size
    x = mx.random.normal((1, rows, HC * hidden)).astype(mx.bfloat16)
    mx.eval(x)
    assert hc_fused.compatible(module, x)
    fused = hc_fused.fused_forward(module, x)
    assert fused is not None
    canonical = module._forward(x)
    mx.eval(fused, canonical)
    ref_mixed, ref_inject = _reference_fp32(module, x)
    if use_combine:
        fused_mixed, passthrough, fused_inject = fused
        canon_mixed, _, canon_inject = canonical
        assert passthrough is x
        assert fused_inject.shape == canon_inject.shape == (1, rows, HC)
        assert _ulps(fused_inject, canon_inject)[0] <= 4
        assert _ulps(fused_inject, ref_inject)[0] <= 4
    else:
        fused_mixed, canon_mixed = fused, canonical
    assert fused_mixed.shape == canon_mixed.shape == (1, rows, hidden)
    assert fused_mixed.dtype == mx.bfloat16
    max_vs_canon, mean_vs_canon = _ulps(fused_mixed, canon_mixed)
    assert max_vs_canon <= 16 and mean_vs_canon <= 0.5
    # Both paths round differently; judge each against fp32. The fused path keeps fp32 through
    # the epilogues, so it must stay at least as close to fp32 as the canonical path (with slack).
    max_fused, mean_fused = _ulps(fused_mixed, ref_mixed)
    max_canon, mean_canon = _ulps(canon_mixed, ref_mixed)
    assert max_fused <= max(2 * max_canon, 6)
    assert mean_fused <= mean_canon * 1.5 + 0.05


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("hidden", [HIDDEN, 768, 1152, 1344])
def test_fused_norm_is_bit_identical_to_rms_norm(hidden):
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(7)
    module = _module(4, hidden=hidden)
    width = HC * hidden
    x = (mx.random.normal((1, 4, width)) * 3).astype(mx.bfloat16)
    flat = x.reshape(4, width)
    normed = hc_fused._kernel(
        "omlx_qwen4_hc_fused_norm", ["x", "w", "eps"], ["xn"], hc_fused._N_SOURCE
    )(
        inputs=[flat, module.hc_norm.weight, hc_fused._eps_array(module)],
        template=[("T", mx.bfloat16), ("K", width), ("H", hidden)],
        grid=(256, HC, 4),
        threadgroup=(256, 1, 1),
        output_shapes=[(4, width)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ]
    expected = module.hc_norm(x).reshape(4, width)
    mx.eval(normed, expected)
    assert mx.array_equal(normed.view(mx.uint16), expected.view(mx.uint16)).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_gated_residual_call_routes_through_fused_path(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.random.normal((1, 2, WIDTH)).astype(mx.bfloat16)
    calls = []
    original = hc_fused.fused_forward
    monkeypatch.setattr(
        hc_fused, "fused_forward", lambda m, h: calls.append(h.shape) or original(m, h)
    )
    out = module(x)
    mx.eval(out)
    assert calls == [(1, 2, WIDTH)]


@pytest.mark.parametrize("hidden,bits", [(800, 4), (1056, 5)])
def test_compatible_rejects_hidden_not_multiple_of_64(hidden, bits):
    # 64 is the quantisation group and the up kernel's grid unit; nothing below it is handled.
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(bits, True, hidden=hidden)
    x = mx.random.normal((1, 4, HC * hidden)).astype(mx.bfloat16)
    assert not hc_fused.compatible(module, x)


def test_ineligible_model_is_logged_once(monkeypatch, caplog):
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_INELIGIBLE_LOGGED", False)
    module = _module(4, True, hidden=800)
    x = mx.random.normal((1, 4, HC * 800)).astype(mx.bfloat16)
    with caplog.at_level("INFO", logger=hc_fused.logger.name):
        assert not hc_fused.compatible(module, x)
        assert not hc_fused.compatible(module, x)
        # Prefill-sized inputs are expected to skip the fused path and must not log.
        monkeypatch.setattr(hc_fused, "_INELIGIBLE_LOGGED", False)
        assert not hc_fused.compatible(
            _module(4), mx.random.normal((1, 64, WIDTH)).astype(mx.bfloat16)
        )
    messages = [
        r.getMessage()
        for r in caplog.records
        if "fused hyper-connection kernels not used" in r.getMessage()
    ]
    assert len(messages) == 1
    assert "hidden_size=800" in messages[0]


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_fused_path_takes_precedence_over_exact_hybrid_projection(monkeypatch):
    # Fused dispatch takes precedence over the compiled hybrid decode path.
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    module._omlx_exact_hybrid_projection = True
    module._compiled_forward = lambda h: pytest.fail(
        "compiled single-token path must not run"
    )
    x = mx.random.normal((1, 1, WIDTH)).astype(mx.bfloat16)
    assert hc_fused.compatible(module, x)
    calls = []
    original = hc_fused.fused_forward
    monkeypatch.setattr(
        hc_fused, "fused_forward", lambda m, h: calls.append(h.shape) or original(m, h)
    )
    out = module(x)
    mx.eval(out)
    assert calls == [(1, 1, WIDTH)]


def test_compatible_fails_closed():
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    ok = mx.random.normal((1, 4, WIDTH)).astype(mx.bfloat16)
    if mx.metal.is_available():
        assert hc_fused.compatible(module, ok)
    assert not hc_fused.compatible(
        module, mx.random.normal((1, 17, WIDTH)).astype(mx.bfloat16)
    )
    assert not hc_fused.compatible(
        module, mx.random.normal((2, 9, WIDTH)).astype(mx.bfloat16)
    )
    assert not hc_fused.compatible(
        module, mx.random.normal((1, 4, WIDTH)).astype(mx.float16)
    )
    assert not hc_fused.compatible(
        module, mx.random.normal((4, WIDTH)).astype(mx.bfloat16)
    )
    module.input_inject_weight = nn.Linear(WIDTH, LOWRANK + HC, bias=False)
    assert not hc_fused.compatible(module, ok)
    del module.input_inject_weight
    module.input_mix_weight_down = nn.Linear(WIDTH, LOWRANK, bias=False)
    assert not hc_fused.compatible(module, ok)
    # The layout verdict is cached per module; replacing a weight tensor must re-check it.
    module = _module(4)
    if mx.metal.is_available():
        assert hc_fused.compatible(module, ok)
    module.input_mix_weight_up.scales = module.input_mix_weight_up.scales.astype(
        mx.float16
    )
    assert not hc_fused.compatible(module, ok)


def test_kill_switch_disables_fused_path(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_HC_FUSED", "0")
    from mlx_vlm.models.qwen4_exp import hc_fused

    reloaded = importlib.reload(hc_fused)
    try:
        assert not reloaded.enabled()
        assert not reloaded.write_enabled()
        assert not reloaded.compatible(
            _module(4), mx.random.normal((1, 4, WIDTH)).astype(mx.bfloat16)
        )
    finally:
        monkeypatch.delenv("OMLX_QWEN4_HC_FUSED")
        importlib.reload(hc_fused)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("source_name", ["_N_SOURCE", "_D_SOURCE", "_U_SOURCE"])
@pytest.mark.parametrize("use_combine", [False, True])
def test_lazy_compilation_failure_returns_canonical_output(
    monkeypatch, caplog, source_name, use_combine
):
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_KERNELS", {})
    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_FAILURE_LOGGED", False)
    monkeypatch.setattr(
        hc_fused,
        source_name,
        getattr(hc_fused, source_name) + "\nintentional_compile_error;\n",
    )
    module = _module(4, use_combine, hidden=64)
    x = mx.ones((1, 1, HC * 64), dtype=mx.bfloat16)
    expected = module._forward(x)
    mx.eval(expected)
    with caplog.at_level("WARNING", logger=hc_fused.logger.name):
        actual = module(x)
        mx.eval(actual)
        repeated = module(x)
        mx.eval(repeated)
    assert not hc_fused._VALIDATED
    assert hc_fused.enabled()
    if not use_combine:
        actual, expected, repeated = [actual], [expected], [repeated]
    for value, reference, again in zip(actual, expected, repeated):
        assert mx.array_equal(value, reference).item()
        assert mx.array_equal(again, reference).item()
    messages = [r for r in caplog.records if "failed closed" in r.getMessage()]
    assert len(messages) == 1
    assert "intentional_compile_error" in messages[0].getMessage()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_specializations_validate_once_and_keep_warm_calls_lazy(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    # Cover each varying template input, including mixed projection bit widths.
    for hidden, rows, down_bits, up_bits, inject_bits in [
        (64, 1, 4, 4, 4),
        (64, 4, 4, 4, 4),
        (128, 4, 4, 4, 4),
        (128, 4, 5, 4, 4),
        (128, 4, 5, 6, 4),
        (128, 4, 5, 6, 8),
        (128, 4, 5, 6, None),
    ]:
        module = _module(down_bits, inject_bits is not None, hidden=hidden)
        module.input_mix_weight_up = _module(up_bits, hidden=hidden).input_mix_weight_up
        if inject_bits is not None:
            module.block_inject_weight = _module(
                inject_bits, hidden=hidden
            ).block_inject_weight
        hc_fused._eps_array(module)
        x = mx.ones((1, rows, HC * hidden), dtype=mx.bfloat16)
        mx.eval(x)
        with monkeypatch.context() as patch:
            evaluate = Mock(wraps=mx.eval)
            patch.setattr(mx, "eval", evaluate)
            first = module(x)
            second = module(x)
            assert evaluate.call_count == 1
        mx.eval(first, second)
    assert len(hc_fused._VALIDATED) == 7


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("rows", [17, 2048])
@pytest.mark.parametrize("use_combine", [True, False])
def test_prefill_path_matches_canonical(bits, rows, use_combine, monkeypatch):
    """Prefill fuses the stream norm and, with inject weights, the mix/inject tail."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(bits, use_combine)
    x = (mx.random.normal((1, rows, WIDTH)) * 2).astype(mx.bfloat16)
    mx.eval(x)
    assert not hc_fused.compatible(module, x)
    assert hc_fused.prefill_compatible(module, x)
    tail = Mock(wraps=hc_fused._tail)
    monkeypatch.setattr(hc_fused, "_tail", tail)
    out = hc_fused.prefill_forward(module, x)
    assert out is not None
    assert tail.called is not use_combine
    canon = module._forward(x)
    if use_combine:
        mixed, passthrough, inject = out
        canon_mixed, _, canon_inject = canon
        assert passthrough is x
        assert inject.shape == canon_inject.shape == (1, rows, HC)
        assert _ulps(inject, canon_inject)[0] <= 4
    else:
        mixed, canon_mixed = out, canon
    assert mixed.shape == canon_mixed.shape == (1, rows, HIDDEN)
    assert mixed.dtype == mx.bfloat16
    max_ulps, mean_ulps = _ulps(mixed, canon_mixed)
    assert max_ulps <= 16 and mean_ulps <= 0.5


# The prefill tail/inject arithmetic as originally shipped: one row per threadgroup, the
# normed row read from device memory for both the mix and the inject dot. Kernel changes
# must keep these bits (prefix caches and accuracy baselines depend on them).
_REFERENCE_TAIL_INJECT = r"""
    const uint row = threadgroup_position_in_grid.z;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const device T* up_r = up + (size_t)row * K;
    const device T* xn_r = xn + (size_t)row * K;
    for (int h = int(t); h < H; h += 256) {
        float acc = 0.0f;
        for (int s = 0; s < HC; ++s) {
            const int n = s * H + h;
            const T g = T(1.0f / (1.0f + metal::exp(-float(up_r[n]))));
            const T p = T(float(g) * float(xn_r[n]));
            acc = s == 0 ? float(p) : float(T(acc + float(p)));
        }
        mixed[(size_t)row * H + h] = T(acc * (1.0f / float(HC)));
    }
    constexpr int PF = hc_pack_factor<BITS_I>();
    constexpr int BP = hc_bytes_per_pack<BITS_I>();
    constexpr int ROW_BYTES = K * BP / PF;
    constexpr int GROUPS = K / 64;
    constexpr int PER = K / 256;
    float res[HC] = {0.0f};
    float xv[PF];
    const int e0 = int(t) * PER;
    for (int e = e0; e < e0 + PER; e += PF) {
        const float sum = hc_load_vector<T, PF, BITS_I>(xn_r + e, xv);
        const int g = e / 64;
        for (int r = 0; r < HC; ++r) {
            res[r] += hc_qdot<PF, BITS_I>(
                (const device uint8_t*)inject_w + r * ROW_BYTES + e * BP / PF,
                xv, float(inject_s[r * GROUPS + g]), float(inject_b[r * GROUPS + g]),
                sum);
        }
    }
    threadgroup float part[HC][8];
    for (int r = 0; r < HC; ++r) {
        const float v = simd_sum(res[r]);
        if (lane == 0) part[r][sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < HC) {
        float v = 0.0f;
        for (int i = 0; i < 8; ++i) v += part[t][i];
        const float q = float(T(float(T(v)) / float(HC)));
        const float gate = float(T(1.0f / (1.0f + metal::exp(-q))));
        inj[(size_t)row * HC + t] = T(2.0f * gate);
    }
"""


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("rows,seed", [(17, 0), (300, 1), (300, 2)])
def test_prefill_tail_inject_keeps_reference_bits(bits, rows, seed):
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(900 + seed * 10 + bits)
    module = _module(bits)
    inject = module.block_inject_weight
    up = (mx.random.normal((rows, WIDTH)) * 3).astype(mx.bfloat16)
    normed = (mx.random.normal((rows, WIDTH)) * 2).astype(mx.bfloat16)

    def run(name, source, header):
        return hc_fused._kernel(
            name,
            ["up", "xn", "inject_w", "inject_s", "inject_b"],
            ["mixed", "inj"],
            source,
            header=header,
        )(
            inputs=[up, normed, inject.weight, inject.scales, inject.biases],
            template=[
                ("T", mx.bfloat16),
                ("BITS_I", bits),
                ("K", WIDTH),
                ("H", HIDDEN),
                ("HC", HC),
            ],
            grid=(256, 1, rows),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, HIDDEN), (rows, HC)],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
        )

    expected = run(
        "test_reference_tail_inject", _REFERENCE_TAIL_INJECT, hc_fused._HEADER
    )
    actual = run(
        "omlx_qwen4_hc_prefill_tail_inject",
        hc_fused._TI_SOURCE,
        hc_fused._HEADER + hc_fused._TG_HEADER,
    )
    mx.eval(expected, actual)
    for observed, reference in zip(actual, expected):
        assert mx.array_equal(observed.view(mx.uint16), reference.view(mx.uint16)).item()


def test_prefill_activation_rounds_like_eager_ops():
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(5)
    special = mx.array(
        [0.0, -0.0, 1e-40, -1e-40, 3e38, -3e38, float("inf"), -float("inf"), float("nan")]
    )
    for scale in (1e-38, 1e-3, 1.0, 40.0, 1e5):
        y = mx.concatenate([mx.random.normal((4096,)) * scale, special]).astype(
            mx.bfloat16
        ).reshape(1, -1, 1)
        expected = nn.silu(y / HC)
        actual = hc_fused._act(HC)(y)
        mx.eval(expected, actual)
        assert mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (1, 16), (1, 17), (1, 2048)]
)
@pytest.mark.parametrize("use_combine", [True, False])
def test_pending_write_matches_eager_write(bits, shape, use_combine):
    """The write-norm kernels store the eager residual bit for bit and normalize the same bits."""
    from mlx_vlm.models.qwen4_exp import hc_fused, language

    mx.random.seed(11 + 31 * bits + 7 * shape[0] + shape[1])
    module = _module(bits, use_combine)
    hyper = (mx.random.normal((*shape, WIDTH)) * 2).astype(mx.bfloat16)
    branch = mx.random.normal((*shape, HIDDEN)).astype(mx.bfloat16)
    gate = (2 * mx.sigmoid(mx.random.normal((*shape, HC)))).astype(mx.bfloat16)
    written = language._hc_write(hyper, branch, gate)
    decode = shape[0] * shape[1] <= hc_fused.MAX_ROWS
    forward = hc_fused.fused_forward if decode else hc_fused.prefill_forward
    expected = forward(module, written)
    actual = forward(module, hyper, (branch, gate))
    expected = (
        (expected[0], written, expected[2]) if use_combine else (expected, written)
    )
    assert actual is not None and len(actual) == len(expected)
    mx.eval(actual, expected)
    for observed, reference in zip(actual, expected):
        assert observed.shape == reference.shape
        assert mx.array_equal(
            observed.view(mx.uint16), reference.view(mx.uint16)
        ).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("rows", [2, 64])
def test_promoting_pending_write_falls_back_to_eager_write(rows):
    # An FP32 branch promotes the eager write; the BF16 kernels must not take it.
    from mlx_vlm.models.qwen4_exp import language

    mx.random.seed(rows)
    module = _module(6)
    hyper = mx.random.normal((1, rows, WIDTH)).astype(mx.bfloat16)
    branch = mx.random.normal((1, rows, HIDDEN))
    gate = (2 * mx.sigmoid(mx.random.normal((1, rows, HC)))).astype(mx.bfloat16)
    expected = module(language._hc_write(hyper, branch, gate))
    actual = module(hyper, write=(branch, gate))
    mx.eval(actual, expected)
    assert actual[1].dtype == mx.float32
    for observed, reference in zip(actual, expected):
        assert mx.array_equal(observed, reference).item()


def test_prefill_path_not_offered_for_fused_rows():
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, hc_fused.MAX_ROWS, WIDTH), dtype=mx.bfloat16)
    assert not hc_fused.prefill_compatible(module, x)


def test_prefill_path_kill_switch(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 64, WIDTH), dtype=mx.bfloat16)
    assert hc_fused.prefill_compatible(module, x)
    monkeypatch.setattr(hc_fused, "_DISABLED", True)
    assert not hc_fused.prefill_compatible(module, x)


def test_module_routes_prefill_rows_through_the_prefill_path(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 64, WIDTH), dtype=mx.bfloat16)
    calls = []
    monkeypatch.setattr(hc_fused, "prefill_forward", lambda m, h: calls.append(h.shape) or "prefill")
    assert module(x) == "prefill"
    assert calls == [(1, 64, WIDTH)]
    assert module(x, target_verify=True) != "prefill"


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("path", ["prefill", "decode"])
def test_transient_failure_preserves_other_models_and_recovers(monkeypatch, path):
    from mlx_vlm.models.qwen4_exp import hc_fused

    failed = _module(4, hidden=64)
    other = _module(8, hidden=64)
    decode = mx.ones((1, 4, HC * 64), dtype=mx.bfloat16)
    inputs = mx.ones((1, 32 if path == "prefill" else 4, HC * 64), dtype=mx.bfloat16)
    expected = failed._forward(inputs)
    mx.eval(expected)
    hook = "_tail" if path == "prefill" else "_kernel_norm"
    with monkeypatch.context() as fault:
        fault.setattr(
            hc_fused, hook, Mock(side_effect=RuntimeError("transient failure"))
        )
        actual = failed(inputs)
        mx.eval(actual)
    for value, reference in zip(actual, expected):
        assert mx.array_equal(value, reference).item()

    assert hc_fused.compatible(other, decode)
    assert hc_fused.compatible(failed, decode)
    if path == "prefill":
        assert hc_fused.prefill_compatible(failed, inputs)
    monkeypatch.setattr(
        failed, "_forward", Mock(side_effect=AssertionError("Unexpected fallback"))
    )
    monkeypatch.setattr(
        other, "_forward", Mock(side_effect=AssertionError("Unexpected fallback"))
    )
    mx.eval(failed(inputs))
    mx.eval(other(decode, target_verify=True))


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("batch,length", [(1, 3), (2, 2), (2, 8)])
def test_fused_rows_match_independent_singletons(bits, batch, length):
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(3770 + bits)
    module = _module(bits)
    x = mx.random.normal((batch, length, WIDTH)).astype(mx.bfloat16)
    actual, _, actual_injection = hc_fused.fused_forward(module, x)
    singletons = [
        hc_fused.fused_forward(module, row.reshape(1, 1, WIDTH))
        for row in x.reshape(-1, WIDTH)
    ]
    expected = mx.concatenate([row[0] for row in singletons], axis=1).reshape(
        batch, length, HIDDEN
    )
    expected_injection = mx.concatenate([row[2] for row in singletons], axis=1).reshape(
        batch, length, HC
    )
    mx.eval(actual, expected, actual_injection, expected_injection)
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(actual_injection, expected_injection).item()
