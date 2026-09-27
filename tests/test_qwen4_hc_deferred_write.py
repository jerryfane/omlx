# SPDX-License-Identifier: Apache-2.0
"""Deferred hyper-connection residual writes: a real-shape Qwen4 stack must match eager writes bit for bit.

Each layer's tail write is carried into the next hyper-connection norm (and the
final mixer). The reference is the same stack with OMLX_QWEN4_HC_FUSED_WRITE off,
which applies every write eagerly as before.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


def _stack(seed: int):
    """Four layers (DeltaNet, DeltaNet+PLE, sparse attention, DeltaNet) at the checkpoint's HC shapes."""
    from mlx_vlm.models import qwen4_exp
    from mlx_vlm.models.qwen4_exp import language

    text = qwen4_exp.TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=2560,
        num_hidden_layers=4,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=10,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=256,
        num_key_value_heads=2,
        max_position_embeddings=65536,
        hc_count=4,
        hc_lowrank=320,
        head_dim=64,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        indexer_budget=64,
        indexer_compress_ratio=4,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [16, 8, 8],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )
    vision = qwen4_exp.VisionConfig(
        model_type="qwen4_exp",
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=32,
        num_heads=4,
        patch_size=14,
        in_channels=3,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    config = qwen4_exp.ModelConfig(
        text_config=text,
        vision_config=vision,
        model_type="qwen4_exp",
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
        vocab_size=256,
    )
    mx.random.seed(seed)
    model = language.LanguageModel(text, config)
    head = language.Qwen4ExpMTPModule(text)
    # Checkpoint layout: 6-bit layer HC with one 8-bit layer, 5-bit final mixer.
    for module, layout in (
        (
            model,
            [
                ("model.layers.3.", 8),
                ("model.layers.", 6),
                ("model.hyper_connection_mixer", 5),
            ],
        ),
        (head, [("", 6)]),
    ):
        module.set_dtype(mx.bfloat16)
        for name, sub in module.named_modules():
            if name.endswith("hc_norm"):
                sub.weight = (mx.random.normal(sub.weight.shape) * 0.05).astype(
                    mx.bfloat16
                )
        for prefix, bits in layout:
            nn.quantize(
                module,
                group_size=64,
                bits=bits,
                class_predicate=lambda path, m, prefix=prefix: isinstance(m, nn.Linear)
                and path.startswith(prefix)
                and "hyper_connection" in path,
            )
        mx.eval(module.parameters())
    return model, head


def _run(monkeypatch, model, head, seed, deferred, script):
    from mlx_vlm.models.qwen4_exp import hc_fused, language

    monkeypatch.setattr(hc_fused, "_WRITE_DISABLED", not deferred)
    eager_writes = []
    real_write = language._hc_write
    monkeypatch.setattr(
        language,
        "_hc_write",
        lambda *args: eager_writes.append(args[0].shape) or real_write(*args),
    )
    rng = np.random.default_rng(seed)
    outputs = script(
        model,
        head,
        lambda batch, length: mx.array(
            rng.integers(2, 256, (batch, length)), dtype=mx.int32
        ),
    )
    mx.eval(outputs)
    return outputs, eager_writes


def _assert_identical(monkeypatch, seed, script):
    model, head = _stack(seed)
    expected, eager = _run(monkeypatch, model, head, seed, False, script)
    actual, deferred = _run(monkeypatch, model, head, seed, True, script)
    # Only the PLE layer still materializes its incoming write.
    assert len(deferred) < len(eager)
    assert len(actual) == len(expected)
    for index, (observed, reference) in enumerate(zip(actual, expected)):
        assert observed.dtype == reference.dtype == mx.bfloat16, index
        assert observed.shape == reference.shape, index
        assert mx.array_equal(
            observed.view(mx.uint16), reference.view(mx.uint16)
        ).item(), f"output {index} differs"


def _decode_and_verify(model, head, ids):
    outputs = []
    cache = model.make_cache()
    outputs.append(model(ids(1, 17), cache=cache).logits)
    for rows in (1, 2, 3, 4):
        outputs.append(model(ids(1, rows), cache=cache).logits)
    for rows in (1, 2, 3, 4):
        # Lightning MTP verify: target-verify rows plus the pre-mixer residual
        # (a one-row window is the decode step itself, with no transaction).
        out = model(ids(1, rows), cache=cache, return_hidden=True)
        hidden = out.hidden_states[0]
        if out.gdn_states is not None:
            out.gdn_states.commit([rows])
        mixed, residual = head(hidden, ids(1, rows), model.model.embed_tokens)
        outputs += [out.logits, hidden, mixed, residual]
    sink = []
    outputs.append(
        model.model(ids(1, 3), cache=cache, hidden_sink=sink, capture_layer_ids=[0, 2])
    )
    outputs += sink
    batch = model.make_cache()
    outputs.append(model(ids(2, 9), cache=batch).logits)
    for rows in (1, 2):
        outputs.append(model(ids(2, rows), cache=batch).logits)
    return outputs


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decode_and_verify_rows_match_eager_writes(monkeypatch, seed):
    _assert_identical(monkeypatch, seed, _decode_and_verify)


@pytest.mark.parametrize("chunk", [17, 257, 2048])
def test_prefill_chunks_match_eager_writes(monkeypatch, chunk):
    def prefill(model, head, ids):
        cache = model.make_cache()
        first = model(ids(1, chunk), cache=cache).logits
        second = model(ids(1, chunk), cache=cache).logits
        return [first, second, model(ids(1, 1), cache=cache).logits]

    _assert_identical(monkeypatch, 40 + chunk, prefill)
