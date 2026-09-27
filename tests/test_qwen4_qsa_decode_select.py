# SPDX-License-Identifier: Apache-2.0
"""One-launch Qwen4 QSA decode selection against the official indexer ops.

An aligned one-row decode (masked-SDPA arm, below the gathered crossover)
used to score the pooled blocks with maximum/sum/divide, pick the winners with
``mx.argpartition`` and widen them to a token mask with ~20 more ops. The
kernel must produce the identical mask: same FP32 scores, the same winners on
ties (MLX's stable ascending sort keeps the highest indices), NaN above +inf,
-0 == +0.
"""

from __future__ import annotations

import importlib
import math

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
qsa_fast = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")

TOPK = 512
RATIO = 4
HEAD_DIM = 128


def _official_mask(head_scores: mx.array, key_len: int) -> mx.array:
    """``Qwen4ExpQSAIndexer.from_projected`` after the matmul, seq_len 1, aligned."""
    batch, seq_len = 1, 1
    past_len = key_len - 1
    max_complete_blocks = key_len // RATIO
    complete_key_len = max_complete_blocks * RATIO
    scores = mx.sum(mx.maximum(head_scores, 0), axis=1)
    scores = scores / math.sqrt(HEAD_DIM)
    query_ends = past_len + mx.arange(seq_len) + 1
    complete_counts = query_ends // RATIO
    valid_blocks = (
        mx.arange(max_complete_blocks)[None, None, :] < complete_counts[None, :, None]
    )
    scores = mx.where(valid_blocks, scores, -mx.inf)
    selected_blocks = mx.argpartition(scores, kth=-TOPK, axis=-1)[..., -TOPK:]
    block_hits = mx.put_along_axis(
        mx.zeros((batch, seq_len, max_complete_blocks), dtype=mx.bool_),
        selected_blocks,
        mx.array(True),
        axis=-1,
    )
    selected_tokens = mx.repeat(block_hits, RATIO, axis=-1)
    if complete_key_len < key_len:
        selected_tokens = mx.concatenate(
            [
                selected_tokens,
                mx.zeros((batch, seq_len, key_len - complete_key_len), dtype=mx.bool_),
            ],
            axis=-1,
        )
    token_indices = mx.arange(key_len)
    tail_starts = complete_counts * RATIO
    tail = (token_indices[None, None, :] >= tail_starts[None, :, None]) & (
        token_indices[None, None, :] < query_ends[None, :, None]
    )
    causal = token_indices[None, None, :] < query_ends[None, :, None]
    use_sparse = complete_counts > TOPK
    selected_tokens = mx.where(use_sparse[None, :, None], selected_tokens | tail, causal)
    return selected_tokens[:, None]


def _head_scores(blocks: int, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "normal":
        return rng.standard_normal((4, blocks)).astype(np.float32)
    if kind == "magnitudes":
        # Sums whose value depends on the order the four heads are added.
        return (
            rng.standard_normal((4, blocks)) * np.exp2(rng.integers(-30, 30, (4, blocks)))
        ).astype(np.float32)
    if kind == "cutoff_ties":
        # 160 strict winners, then a tie that straddles the top-512 cutoff.
        scores = np.full((4, blocks), -1.0, dtype=np.float32)
        order = rng.permutation(blocks)
        tied = min(1400, blocks - 160)
        scores[:, order[:tied]] = 0.5
        scores[:, order[tied : tied + 160]] = 2.0
        return scores
    if kind == "specials":
        scores = rng.standard_normal((4, blocks)).astype(np.float32)
        for value, count in ((np.nan, 150), (np.inf, 150), (-np.inf, 150), (-0.0, 600)):
            scores[rng.integers(0, 4, count), rng.integers(0, blocks, count)] = value
        return scores
    raise ValueError(kind)


@pytest.mark.parametrize(
    "key_len",
    [
        2052,  # 513 blocks: one above the budget
        2055,  # three-token tail
        24003,  # 24K decode
        32768,  # 8192 blocks: largest bank held 8 per thread
        32773,  # 8193 blocks, one-token tail: 16 per thread
    ],
)
@pytest.mark.parametrize("kind", ["normal", "magnitudes", "cutoff_ties", "specials"])
def test_decode_mask_matches_the_official_indexer_ops(key_len, kind):
    blocks = key_len // RATIO
    head_scores = mx.array(_head_scores(blocks, kind, key_len).reshape(1, 4, 1, blocks))

    actual = qsa_fast.decode_block_selection_mask(
        head_scores,
        head_dim=HEAD_DIM,
        key_tokens=key_len,
        compress_ratio=RATIO,
        block_topk=TOPK,
    )
    expected = _official_mask(head_scores, key_len)

    assert actual is not None
    assert actual.shape == expected.shape == (1, 1, 1, key_len)
    assert actual.dtype == mx.bool_
    assert mx.array_equal(actual, expected).item()


def test_cutoff_tie_keeps_the_highest_block_indices():
    blocks, key_len = 2048, 8192
    scores = np.full((4, 1, blocks), -1.0, dtype=np.float32)
    strict = np.arange(0, 100)
    tied = np.arange(300, 1500)
    scores[0, 0, strict] = 3.0
    scores[0, 0, tied] = 1.0
    mask = qsa_fast.decode_block_selection_mask(
        mx.array(scores[None]),
        head_dim=HEAD_DIM,
        key_tokens=key_len,
        compress_ratio=RATIO,
        block_topk=TOPK,
    )
    selected = np.flatnonzero(np.asarray(mask).reshape(-1)[::RATIO])
    expected = np.concatenate((strict, tied[-(TOPK - strict.size) :]))
    np.testing.assert_array_equal(selected, expected)


def test_block_scores_round_like_maximum_sum_divide():
    """The kernel's score is bit-identical to MLX's three ops, NaN payloads included."""
    kernel = mx.fast.metal_kernel(
        name="test_qwen4_qsa_decode_block_score",
        input_names=["head_scores", "divisor"],
        output_names=["out"],
        header=qsa_fast._DECODE_SELECT_HEADER,
        source="""
            const uint n = uint(head_scores_shape[head_scores_ndim - 1]);
            const uint e = thread_position_in_grid.x;
            if (e < n) {
                out[e] = qsa_decode_block_score(head_scores, n, e, H, divisor[0]);
            }
        """,
    )
    blocks = 6000
    for kind in ("magnitudes", "specials"):
        head_scores = mx.array(_head_scores(blocks, kind, 7).reshape(1, 4, 1, blocks))
        expected = mx.sum(mx.maximum(head_scores, 0), axis=1) / math.sqrt(HEAD_DIM)
        actual = kernel(
            inputs=[head_scores, mx.array([math.sqrt(HEAD_DIM)], dtype=mx.float32)],
            template=[("H", 4)],
            grid=(blocks, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, 1, blocks)],
            output_dtypes=[mx.float32],
        )[0]
        assert mx.array_equal(actual.view(mx.uint32), expected.view(mx.uint32)).item()


def _attention_and_caches(key_len: int):
    from mlx_vlm.models.qwen4_exp import TextConfig

    config = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=65536,
        head_dim=64,
        layer_types=["full_attention"],
        ple_layer_ids=[],
        ple_embed_dim=32,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=HEAD_DIM,
        indexer_budget=TOPK * RATIO,
        indexer_compress_ratio=RATIO,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [3, 3, 2],
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        },
    )
    mx.random.seed(3)
    attention = language.Qwen4ExpAttention(config)
    mx.eval(attention.parameters())
    keys = mx.random.normal((1, 2, key_len, 64)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, key_len, 64)).astype(mx.bfloat16)
    index_keys = mx.random.normal((1, key_len, HEAD_DIM)).astype(mx.bfloat16)
    positions = mx.arange(key_len, dtype=mx.int32)[None]
    caches = []
    for _ in range(2):
        cache = language.QSAKVCache()
        cache.update_and_fetch(keys, values)
        cache.update_indexer(index_keys, positions)
        caches.append(cache)
    mx.eval([c.state for c in caches])
    return config, attention, caches


def test_decode_steps_match_the_ops_path_bit_for_bit(monkeypatch):
    """Rank-three (served below the gathered crossover) decode through the
    module: outputs and every cache array equal, across block completions."""
    config, attention, (fast_cache, ops_cache) = _attention_and_caches(3001)
    kernel_masks = []

    def recording(*args, **kwargs):
        mask = qsa_fast.decode_block_selection_mask(*args, **kwargs)
        if mask is not None:
            kernel_masks.append(mask)
        return mask

    monkeypatch.setattr(language, "decode_block_selection_mask", recording)
    for step in range(6):
        x = mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
        offset = fast_cache.offset
        positions = mx.broadcast_to(mx.array([[offset]], dtype=mx.int32)[None], (3, 1, 1))
        monkeypatch.setattr(qsa_fast, "_DECODE_SELECT_DISABLED", False)
        fast = attention(x, mask=None, cache=fast_cache, position_ids=positions)
        monkeypatch.setattr(qsa_fast, "_DECODE_SELECT_DISABLED", True)
        ops = attention(x, mask=None, cache=ops_cache, position_ids=positions)
        mx.eval(fast, ops)
        assert mx.array_equal(fast.view(mx.uint16), ops.view(mx.uint16)).item(), step
    assert len(kernel_masks) == 6
    for fast_state, ops_state in zip(fast_cache.state, ops_cache.state):
        assert mx.array_equal(fast_state, ops_state).item()
    pooled = fast_cache._pooled_index_offset
    assert pooled == ops_cache._pooled_index_offset == 3007 // RATIO
    assert mx.array_equal(
        fast_cache._pooled_index_keys[:, :pooled], ops_cache._pooled_index_keys[:, :pooled]
    ).item()
