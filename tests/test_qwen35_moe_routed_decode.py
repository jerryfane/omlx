# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness and routing tests for the fused one-token routed experts."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import omlx.patches.qwen35_moe_routed_decode as routed

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

EXPERTS = 32


class _FakeQwen4Model:
    pass


_FakeQwen4Model.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"


@pytest.fixture(autouse=True)
def _patched_block(monkeypatch):
    """Apply router + routed patches, restore the class afterwards."""
    from mlx_vlm.models.qwen3_5_moe import language as vlm_moe

    from omlx.patches.qwen35_moe_router import apply_qwen35_moe_router_patch

    cls = vlm_moe.Qwen3_5MoeSparseMoeBlock
    apply_qwen35_moe_router_patch()  # process-wide and idempotent
    assert cls._omlx_router_fused
    call = cls.__call__
    original = getattr(call, "_omlx_routed_decode_original", call)
    monkeypatch.setattr(routed, "_DISABLED", False)
    monkeypatch.setattr(routed, "_PROVEN", False)
    cls.__call__ = original
    if "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")
    assert routed.apply_qwen35_moe_routed_decode_patch()
    yield cls
    cls.__call__ = original
    cls._omlx_router_fused = True
    if "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")


def _block(hidden, inter, top_k=10, bits=4, group_size=64, seed=0, experts=EXPERTS):
    """A block laid out like Qwen3.8-Flash-Next oQ: quantized routed experts,
    8-bit gs128 shared expert, 8-bit gs64 shared-expert gate, bf16 router."""
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

    mx.random.seed(seed)
    args = SimpleNamespace(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=inter,
        num_experts=experts,
        num_experts_per_tok=top_k,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    sm = block.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(sm, name, getattr(sm, name).to_quantized(group_size, bits))
    shared = block.shared_expert
    shared_gs = 128 if hidden % 128 == 0 and inter % 128 == 0 else 64
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(shared, name, nn.QuantizedLinear.from_linear(getattr(shared, name), shared_gs, 8))
    block.shared_expert_gate = nn.QuantizedLinear.from_linear(block.shared_expert_gate, 64, 8)
    block.eval()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("mlp.switch_mlp", sm)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    mx.eval(block.parameters())
    return block


def _pair(block, x):
    routed._DISABLED = True
    ref = block(x)
    mx.eval(ref)
    routed._DISABLED = False
    out = block(x)
    mx.eval(out)
    return ref, out


def _same_bits(a, b):
    return a.shape == b.shape and mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


@pytest.mark.parametrize(
    "hidden,inter,bits,group_size",
    [
        (2560, 640, 5, 64),  # Qwen3.8-Flash-Next oQ5e
        (2560, 640, 4, 64),
        (1024, 320, 5, 32),
        (1024, 384, 6, 128),
        (1024, 320, 8, 64),
    ],
)
def test_fused_decode_is_bit_identical(hidden, inter, bits, group_size, monkeypatch):
    calls = []
    fused = routed.routed_decode
    monkeypatch.setattr(
        routed, "routed_decode", lambda *a: calls.append(1) or fused(*a)
    )
    for seed in range(2):
        block = _block(hidden, inter, bits=bits, group_size=group_size, seed=seed)
        for step in range(6):
            x = (mx.random.normal((1, 1, hidden)) * (0.5 + step)).astype(mx.bfloat16)
            assert routed.routed_decode_plan(block, x) is not None
            ref, out = _pair(block, x)
            assert _same_bits(ref, out)
    assert len(calls) == 12
    assert not routed._DISABLED


@pytest.mark.parametrize("bits", [4, 5])
def test_fp32_kernels_match_mlx_mat_vecs(bits):
    """BF16 outputs hide one-ulp FP32 differences, so run both kernels in
    FP32 against MLX's FP32 gather_qmm: the gate+up rows after SwiGLU, and
    each expert's down rows (read one at a time through the combine with a
    one-hot score and a zero shared gate)."""
    from mlx_vlm.models.activations import swiglu

    hidden, inter, gs, top_k = 2560, 640, 64, routed.TOP_K
    mx.random.seed(40 + bits)
    gu_w, gu_s, gu_b = mx.quantize(
        mx.random.normal((EXPERTS, 2 * inter, hidden)) * 0.05, gs, bits
    )
    d_w, d_s, d_b = mx.quantize(mx.random.normal((EXPERTS, hidden, inter)) * 0.05, gs, bits)
    zero = mx.zeros((hidden,), mx.float32)
    off = mx.array([-1e30], mx.float32)  # sigmoid(off) == 0
    for step in range(3):
        x = mx.random.normal((1, 1, hidden)) * (0.5 + step)
        ids = mx.random.permutation(EXPERTS)[:top_k].astype(mx.uint32)
        ref = mx.gather_qmm(
            mx.expand_dims(x, (-2, -3)), gu_w, gu_s, gu_b, rhs_indices=ids.reshape(1, 1, top_k),
            transpose=True, group_size=gs, bits=bits, sorted_indices=False,
        )
        gate, up = mx.split(ref, 2, axis=-1)
        ref_h = swiglu(gate, up).reshape(top_k, inter)
        h = routed._gate_up_kernel(bits, gs)(
            inputs=[x, gu_w, gu_s, gu_b, ids],
            template=[("T", mx.float32), ("K", hidden), ("N", 2 * inter), ("RPS", 2), ("NSG", 2)],
            grid=(32, inter // 2, top_k),
            threadgroup=(32, 2, 1),
            output_shapes=[(top_k, inter)],
            output_dtypes=[mx.float32],
        )[0]
        assert mx.array_equal(h.view(mx.uint32), ref_h.view(mx.uint32)).item()
        ref_d = mx.gather_qmm(
            mx.expand_dims(h, -2), d_w, d_s, d_b, rhs_indices=ids, transpose=True,
            group_size=gs, bits=bits, sorted_indices=False,
        ).reshape(top_k, hidden)
        for j in range(top_k):
            one_hot = (mx.arange(top_k) == j).astype(mx.float32)
            row = routed._down_kernel(bits, gs)(
                inputs=[zero, off, h, d_w, d_s, d_b, ids, one_hot],
                template=[("T", mx.float32), ("K", inter), ("N", hidden), ("RPS", 4)],
                grid=(32, top_k * hidden // 4, 1),
                threadgroup=(32, top_k, 1),
                output_shapes=[(hidden,)],
                output_dtypes=[mx.float32],
            )[0]
            # The +0 folds of the k-sum turn -0 into +0, so compare values.
            assert mx.array_equal(row, ref_d[j]).item()


def test_experts_past_the_bound_view_are_read_from_the_stacked_weights():
    """The kernels bind a one-expert view and index the rest; a view copied
    out of the stacked buffer would read the wrong bytes for high experts."""
    from omlx.patches.qwen35_moe_router import fused_moe_combine

    block = _block(1024, 320, bits=5, experts=512)
    plan = routed.routed_decode_plan(block, mx.zeros((1, 1, 1024), mx.bfloat16))
    assert all(o.shape[0] == 1 for o in plan.gate_up_operands + plan.down_operands)
    for ids in ([502 + i for i in range(10)], [511, 0, 500, 256, 1, 510, 3, 499, 7, 400]):
        x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
        inds = mx.array(ids, dtype=mx.uint32).reshape(1, 1, 10)
        scores = mx.softmax(mx.random.normal((1, 1, 10)), axis=-1).astype(mx.bfloat16)
        shared = block["shared_expert"](x)
        gate = block["shared_expert_gate"](x)
        ref = fused_moe_combine(block.switch_mlp(x, inds), scores, shared, gate)
        out = routed.routed_decode(plan, x, inds, scores, shared, gate)
        mx.eval(ref, out)
        assert _same_bits(ref, out)


def test_plan_follows_replaced_expert_weights():
    block = _block(1024, 320, bits=5)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    first = routed.routed_decode_plan(block, x)
    donor = _block(1024, 320, bits=5, seed=7)
    for name in ("gate_up_proj", "down_proj"):
        for key in ("weight", "scales", "biases"):
            block.switch_mlp[name][key] = donor.switch_mlp[name][key]
    assert routed.routed_decode_plan(block, x) is not first
    ref, out = _pair(block, x)
    assert _same_bits(ref, out)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden": 1024, "inter": 512},  # down would take qmv_fast
        {"hidden": 960, "inter": 320},  # gate+up would take qmv
        {"hidden": 1024, "inter": 320, "top_k": 8},
        {"hidden": 1024, "inter": 320, "bits": 3},
    ],
)
def test_ineligible_shapes_keep_the_composed_body(kwargs):
    block = _block(**kwargs)
    hidden = kwargs["hidden"]
    x = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16)
    assert routed.routed_decode_plan(block, x) is None


def test_prefill_verify_and_float_rows_keep_the_composed_body():
    block = _block(1024, 320)
    assert routed.routed_decode_plan(block, mx.zeros((1, 4, 1024), dtype=mx.bfloat16)) is None
    assert routed.routed_decode_plan(block, mx.zeros((2, 1, 1024), dtype=mx.bfloat16)) is None
    assert routed.routed_decode_plan(block, mx.zeros((1, 1, 1024), dtype=mx.float16)) is None
    x = mx.random.normal((1, 5, 1024)).astype(mx.bfloat16)
    ref, out = _pair(block, x)
    assert _same_bits(ref, out)


def test_block_without_gate_up_fusion_is_not_routed():
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    args = SimpleNamespace(
        hidden_size=1024,
        moe_intermediate_size=320,
        shared_expert_intermediate_size=320,
        num_experts=EXPERTS,
        num_experts_per_tok=10,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    x = mx.zeros((1, 1, 1024), dtype=mx.bfloat16)
    assert routed.routed_decode_plan(block, x) is None


def test_kernel_failure_falls_back_once(monkeypatch):
    block = _block(1024, 320)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    routed._DISABLED = True
    ref = block(x)
    routed._DISABLED = False

    def broken(*args):
        raise RuntimeError("no pipeline")

    monkeypatch.setattr(routed, "routed_decode", broken)
    out = block(x)
    mx.eval(ref, out)
    assert _same_bits(ref, out)
    assert routed._DISABLED
    assert routed.routed_decode_plan(block, x) is None


def test_apply_requires_the_fused_router(_patched_block):
    cls = _patched_block
    del cls._omlx_routed_decode
    cls._omlx_router_fused = False
    assert not routed.apply_qwen35_moe_routed_decode_patch()


def test_apply_is_idempotent(_patched_block):
    cls = _patched_block
    call = cls.__call__
    assert routed.apply_qwen35_moe_routed_decode_patch()
    assert cls.__call__ is call
