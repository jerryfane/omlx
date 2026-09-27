# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused Qwen MoE router top-k.

The routed expert SET must match the composed chain exactly — including
on equal rounded probabilities, where mlx's ``argpartition`` keeps the
HIGHEST index (pinned here so an mlx behavior change fails loudly).
Scores may differ from the composed sum/divide by reduction-order ulp.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx.patches.qwen35_moe_router import fused_router_topk, router_eligible

K = 8
NE = 256


def _composed(p, top_k=K):
    inds = mx.argpartition(p, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(p, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("experts,top_k", [(NE, K), (512, 10)])
def test_selected_set_matches_composed(experts, top_k):
    mx.random.seed(3)
    for _ in range(300):
        g = (mx.random.normal((1, 1, experts)) * 2.0).astype(mx.bfloat16)
        p = mx.softmax(g, axis=-1, precise=True)
        ci, cs = _composed(p, top_k)
        fi, fs = fused_router_topk(p, top_k)
        assert sorted(ci[0, 0].tolist()) == sorted(fi[0, 0].tolist())
        cmap = dict(zip(ci[0, 0].tolist(), cs[0, 0].astype(mx.float32).tolist()))
        fmap = dict(zip(fi[0, 0].tolist(), fs[0, 0].astype(mx.float32).tolist()))
        for idx, ref in cmap.items():
            assert abs(ref - fmap[idx]) <= max(2e-2 * ref, 1e-4)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_tie_break_matches_argpartition_highest_index():
    base = np.full(NE, 0.001, dtype=np.float32)
    for t in range(0, NE, 16):
        base[t] = 0.1
    for arr in (base, base[::-1].copy()):
        p = mx.array(arr, dtype=mx.bfloat16).reshape(1, 1, -1)
        ci, _ = _composed(p)
        fi, _ = fused_router_topk(p, K)
        assert sorted(ci[0, 0].tolist()) == sorted(fi[0, 0].tolist())


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_multi_row_verify_widths():
    mx.random.seed(5)
    for rows in (1, 4, 6):
        g = (mx.random.normal((1, rows, NE)) * 2.0).astype(mx.bfloat16)
        p = mx.softmax(g, axis=-1, precise=True)
        ci, _ = _composed(p)
        fi, _ = fused_router_topk(p, K)
        for r in range(rows):
            assert sorted(ci[0, r].tolist()) == sorted(fi[0, r].tolist())


def test_eligibility_gates():
    x1 = mx.zeros((1, 1, 2048), dtype=mx.bfloat16)
    assert router_eligible(x1, NE)
    xp = mx.zeros((1, 2048, 2048), dtype=mx.bfloat16)
    assert not router_eligible(xp, NE)  # prefill rows stay composed
    assert not router_eligible(x1, 250)  # NE % 32 != 0
    xf = mx.zeros((1, 1, 2048), dtype=mx.float32)
    assert not router_eligible(xf, NE)


@pytest.mark.parametrize("length,engaged", [(4, True), (9, False)])
def test_verifier_routes_short_moe_blocks_through_fused_router(
    monkeypatch, length, engaged
):
    from types import SimpleNamespace

    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
    from omlx.patches import qwen35_moe_router as router

    monkeypatch.setattr(
        Qwen3_5BatchInvariantForward,
        "_feed_forward",
        Qwen3_5BatchInvariantForward._feed_forward,
    )
    model = Qwen3_5MoeSparseMoeBlock(
        SimpleNamespace(
            hidden_size=64,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_experts=512,
            num_experts_per_tok=10,
        )
    )
    model.set_dtype(mx.bfloat16)
    x = mx.random.normal((1, length, 64)).astype(mx.bfloat16)
    calls = []
    original = router.fused_router_topk

    def record(probs, top_k):
        calls.append((probs.shape, top_k))
        return original(probs, top_k)

    monkeypatch.setattr(router, "fused_router_topk", record)
    router._ensure_vlm_verify_patch()
    router._ensure_vlm_verify_patch()
    mx.eval(Qwen3_5BatchInvariantForward()._feed_forward(model, x))
    assert calls == ([((1, length, 512), 10)] if engaged else [])


def _composed_combine(routed, scores, shared, gate):
    y = (routed * scores[..., None]).sum(axis=-2)
    shared = mx.sigmoid(gate) * shared
    return y + shared


def _same_bits(a, b):
    # NaN payloads aside, every element must carry the same BF16 bits.
    both_nan = mx.isnan(a) & mx.isnan(b)
    return not mx.any((a.view(mx.uint16) != b.view(mx.uint16)) & ~both_nan).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("top_k,hidden", [(10, 2560), (8, 2048)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_combine_matches_composed_ops(top_k, hidden, seed):
    from omlx.patches.qwen35_moe_router import fused_moe_combine

    mx.random.seed(seed)
    routed = (mx.random.normal((1, 1, top_k, hidden)) * (1 + 4 * seed)).astype(
        mx.bfloat16
    )
    special = mx.array([float("inf"), -float("inf"), -0.0, 0.0, 3e38], mx.bfloat16)
    routed[0, 0, 0, : special.size] = special
    scores = mx.softmax(mx.random.normal((1, 1, top_k)), axis=-1).astype(mx.bfloat16)
    shared = (mx.random.normal((1, 1, hidden)) * 2).astype(mx.bfloat16)
    # Gate logits cover both sigmoid branches and exp's rounding ties (-6.84375).
    for logit in (-6.84375, -20.0, -0.0, 0.3, 9.5):
        gate = mx.array([[[logit]]], mx.bfloat16)
        expected = _composed_combine(routed, scores, shared, gate)
        actual = fused_moe_combine(routed, scores, shared, gate)
        assert actual is not None
        mx.eval(expected, actual)
        assert _same_bits(actual, expected)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_fused_combine_sigmoid_matches_every_bf16_gate():
    from omlx.patches.qwen35_moe_router import fused_moe_combine

    mx.random.seed(9)
    routed = mx.random.normal((1, 1, 10, 64)).astype(mx.bfloat16)
    scores = mx.softmax(mx.random.normal((1, 1, 10)), axis=-1).astype(mx.bfloat16)
    shared = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
    gates = mx.arange(0, 65536, dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16)
    for start in range(0, gates.size, 2048):
        chunk = [gates[i].reshape(1, 1, 1) for i in range(start, start + 2048)]
        actual = mx.stack([fused_moe_combine(routed, scores, shared, g) for g in chunk])
        expected = mx.stack([_composed_combine(routed, scores, shared, g) for g in chunk])
        mx.eval(actual, expected)
        assert _same_bits(actual, expected)


def test_fused_combine_declines_other_layouts(monkeypatch):
    from omlx.patches import qwen35_moe_router as router

    def operands(rows=1, top_k=10, dtype=mx.bfloat16):
        return (
            mx.zeros((1, rows, top_k, 64), dtype),
            mx.zeros((1, rows, top_k), dtype),
            mx.zeros((1, rows, 64), dtype),
            mx.zeros((1, rows, 1), dtype),
        )

    assert router.fused_moe_combine(*operands(rows=2)) is None  # verify rows stay composed
    assert router.fused_moe_combine(*operands(top_k=7)) is None
    assert router.fused_moe_combine(*operands(dtype=mx.float16)) is None
    monkeypatch.setattr(router, "_COMBINE_DISABLED", True)
    assert router.fused_moe_combine(*operands()) is None


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("seed", [0, 1])
def test_one_row_moe_block_matches_composed_combine(monkeypatch, seed):
    """The patched one-row decode block (5-bit experts, 8-bit shared expert) keeps its bits."""
    from types import SimpleNamespace

    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
    from omlx.patches import qwen35_moe_router as router

    assert router.apply_qwen35_moe_router_patch()
    mx.random.seed(seed)
    block = Qwen3_5MoeSparseMoeBlock(
        SimpleNamespace(
            hidden_size=2560,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=128,
            num_experts=32,
            num_experts_per_tok=10,
        )
    )
    block.set_dtype(mx.bfloat16)
    sm = block.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(sm, name, getattr(sm, name).to_quantized(64, 5))
    for name in ("gate_proj", "up_proj", "down_proj"):
        linear = getattr(block.shared_expert, name)
        setattr(block.shared_expert, name, nn.QuantizedLinear.from_linear(linear, 128, 8))
    block.shared_expert_gate = nn.QuantizedLinear.from_linear(block.shared_expert_gate, 64, 8)
    mx.eval(block.parameters())
    calls = []
    real = router.fused_moe_combine
    monkeypatch.setattr(
        router, "fused_moe_combine", lambda *a: calls.append(a[0].shape) or real(*a)
    )
    x = mx.random.normal((1, 1, 2560)).astype(mx.bfloat16)
    with monkeypatch.context() as off:
        off.setattr(router, "_COMBINE_DISABLED", True)
        expected = block(x)
        mx.eval(expected)
    actual = block(x)
    mx.eval(actual)
    assert calls == [(1, 1, 10, 2560)] * 2
    assert _same_bits(actual, expected)
