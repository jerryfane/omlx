#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Row-exact Lightning MTP verify diagnostic for Qwen4 (Qwen3.8-Flash-Next).

With Lightning MTP on, every row of a single-stream verify window must produce
the bits of the one-row serial decode step at its position, so greedy MTP-on
output equals MTP-off output. This script checks that promise component by
component on the GPU it runs on, with synthetic weights at the real
Qwen3.8-Flash-Next shapes (no model download, no model load).

Usage, from the omlx repository root with the Python that serves oMLX::

    python scripts/diag_row_exact.py              # everything (a few minutes)
    python scripts/diag_row_exact.py --quick      # fewer rows/contexts
    python scripts/diag_row_exact.py --only qmv,sdpa,gdn
    python scripts/diag_row_exact.py --list       # section names

Sections (each prints one line per check: PASS / FAIL / WARN / INFO / ERROR):

  env       device architecture, MLX / macOS versions, NAX, OMLX_* / MLX_* env
  sdpa      the SDPA plan mirror: MLX's one-row vector SDPA at N keys vs the
            row-exact verify chunks (_row_exact_causal_sdpa) at key counts
            crossing 1024 / 4096 / 8192 / 16384 / 65536, and the QSA masked
            decode kernel (a JIT copy of MLX's two-pass kernel with the
            mirrored partition count) vs MLX itself
  qmv       stock metallib quantized_matmul (what a serial step runs) vs the
            JIT row_exact_qmv transcription (what a verify row runs), per
            served projection: attention q/k/v/o/index_qk, PLE key/value,
            lm_head, the 8-bit GDN layer's stacked in-projection and its
            out-projection, and the RowsQmv / OneRowQmv tile geometries; plus
            the verify kernel's own one-row tile run once per row vs the verify
            rows (JIT vs JIT: whether a serial step sharing the verify pipeline
            would restore equality where stock vs JIT fails)
  gdn       whole GatedDeltaNet layer: serial one-token steps vs the fused
            verify window, for the 6-bit layout (serial = JIT decode plan), the
            checkpoint's 8-bit layer (serial = stock mlx-vlm chain) and that
            layer with the decode plan admitting it (serial = JIT decode plan)
  hc        hyper-connection modules through the served entry: one-row calls
            vs R-row verify calls (R = 1..8, V2 and V1 kernels), with and
            without a pending residual write; reports fused-path fallbacks
  moe       real-shape MoE block (512 experts, top-10): fused one-token decode
            per row vs the row-exact verify window
  attn      full-attention layer (24/2 heads x 256, QSA indexer): serial
            steps vs verify rows and cache state, dense arm (700/1020/1022
            keys) and the masked arm past the 2048-token budget
  ple       the PLE depthwise conv kernel (first-use validation) and whether
            its fallback mx.conv1d is row-exact
  flags     fail-closed fallback flags and captured warnings after the run

A FAIL in "qmv ... stock vs verify JIT" means this GPU's JIT-compiled copy of
qmv_fast does not reproduce the metallib kernel: every verify row of that
projection then differs from the serial step. A FAIL in "gdn"/"hc"/"moe"/"attn" with "qmv" passing
points at that component's own kernels. Exit status is 1 if anything FAILs.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

HIDDEN = 2560
RESULTS: list[tuple[str, str, str]] = []
WARNINGS: list[str] = []


class _Capture(logging.Handler):
    def emit(self, record):
        WARNINGS.append(f"{record.name}: {record.getMessage()}")


def report(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    print(f"[{status:5}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def same_bits(a: mx.array, b: mx.array) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {mx.float32: mx.uint32, mx.bfloat16: mx.uint16, mx.float16: mx.uint16}.get(a.dtype)
    if view is None:
        return bool(mx.array_equal(a, b).item())
    return bool(mx.array_equal(a.view(view), b.view(view)).item())


def diff(a: mx.array, b: mx.array) -> str:
    if a.shape != b.shape or a.dtype != b.dtype:
        return f"shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    neq = mx.not_equal(a32, b32)
    count = int(neq.sum().item())
    worst = float(mx.max(mx.abs(a32 - b32)).item())
    return f"{count}/{a.size} elements differ, max |diff| {worst:.3g}"


def check(name: str, got: mx.array, want: mx.array) -> bool:
    mx.eval(got, want)
    if same_bits(got, want):
        report("PASS", name)
        return True
    report("FAIL", name, diff(got, want))
    return False


def guarded(section):
    def run(args):
        try:
            section(args)
        except Exception as exc:  # noqa: BLE001 - a diagnostic reports, never stops
            report("ERROR", section.__name__, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    run.__name__ = section.__name__
    return run


def patches():
    """The served Qwen4 patch chain (as omlx/engine/vlm.py installs it)."""
    from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from omlx.patches import qwen35_gdn_prework as prework
    from omlx.patches import qwen35_moe_routed_decode as routed
    from omlx.patches import qwen35_moe_router as router
    from omlx.patches import qwen35_verify_qmm
    from omlx.patches.mlx_vlm_mtp import qwen35_verify_linear
    from omlx.patches.qwen35_verify_sdpa_split import apply_qwen35_verify_sdpa_split_patch

    qwen35_verify_qmm.apply_verify_qmm_patch()
    qwen35_verify_linear.apply()
    prework.apply_qwen35_gdn_prework_patch()
    apply_qwen35_verify_sdpa_split_patch()
    router.apply_qwen35_moe_router_patch()
    routed.apply_qwen35_moe_routed_decode_patch()


class armed:
    """Row-exact verify arming, as batch_generator sets it for a B=1 window."""

    def __enter__(self):
        from omlx.patches import qwen35_verify_qmm

        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)

    def __exit__(self, *exc):
        from omlx.patches import qwen35_verify_qmm

        qwen35_verify_qmm.set_verify_qmm_armed(False)


def quantized_linear(k: int, n: int, bits: int, group_size: int, seed: int):
    mx.random.seed(seed)
    linear = nn.QuantizedLinear(k, n, bias=False, group_size=group_size, bits=bits)
    weight = (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16)
    linear.weight, linear.scales, linear.biases = mx.quantize(
        weight, group_size=group_size, bits=bits
    )
    mx.eval(linear.parameters())
    return linear


def stock_rows(linear, x):
    """One stock metallib quantized_matmul per row: the serial decode call."""
    return mx.concatenate(
        [
            mx.quantized_matmul(
                x[:, r : r + 1],
                linear.weight,
                linear.scales,
                linear.biases,
                transpose=True,
                group_size=linear.group_size,
                bits=linear.bits,
                mode="affine",
            )
            for r in range(x.shape[1])
        ],
        axis=1,
    )


def jit_rows(linear, x):
    """The verify kernel's one-row tile (ROWS=1, RPS=4) launched once per row:
    what a serial step would run if it used the verify pipeline."""
    from omlx.patches import row_exact_qmv

    flat = x.reshape(-1, x.shape[-1])
    if not row_exact_qmv._kernel_supported(linear, flat[:1]):
        return None
    plan = row_exact_qmv._Plan(linear, flat[:1], 1)
    rows = [
        plan.kernel(
            inputs=[flat[r : r + 1], linear.weight, linear.scales, linear.biases],
            template=plan.template,
            grid=plan.grid,
            threadgroup=(32, 2, 1),
            output_shapes=plan.output_shapes,
            output_dtypes=plan.output_dtypes,
        )[0]
        for r in range(flat.shape[0])
    ]
    return mx.concatenate(rows, axis=0).reshape(*x.shape[:-1], -1)


# ---------------------------------------------------------------------------


@guarded
def env(args):
    info = mx.device_info()
    arch = str(info.get("architecture", ""))
    report("INFO", "device", f"{info.get('device_name')} architecture={arch}")
    try:
        from omlx.custom_kernels.nax import is_nax_available

        nax = bool(is_nax_available())
    except Exception as exc:  # noqa: BLE001
        nax = f"unknown ({exc})"
    report(
        "INFO",
        "versions",
        f"mlx {mx.__version__} ({Path(mx.__file__).parent}), macOS {platform.mac_ver()[0]}, "
        f"python {platform.python_version()}, NAX {nax}",
    )
    try:
        rev = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        rev = "?"
    report("INFO", "omlx", f"{REPO} @ {rev or '?'}")
    env_vars = {k: v for k, v in os.environ.items() if k.startswith(("OMLX_", "MLX_"))}
    report("INFO", "env", ", ".join(f"{k}={v}" for k, v in sorted(env_vars.items())) or "none set")
    if "MLX_SDPA_BLOCKS" in env_vars:
        report(
            "FAIL",
            "env MLX_SDPA_BLOCKS",
            "MLX overrides its two-pass partition count with it; the row-exact SDPA mirrors do not",
        )


@guarded
def sdpa(args):
    from omlx.patches import qwen35_verify_sdpa_split as split
    from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch

    apply_mlx_vlm_qwen4_exp_compat_patch()
    qsa_fast = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")

    heads, kv_heads, dim = 24, 2, 256
    gqa = heads // kv_heads
    limit = split._VECTOR_ROW_BUDGET // gqa
    scale = dim**-0.5
    report("INFO", "sdpa class", f"arch letter {split._gpu_class()!r}, verify chunk limit {limit} rows")
    contexts = (
        [700, 1023, 1024, 4096, 16384, 65536]
        if args.quick
        else [700, 1022, 1023, 1024, 1025, 4095, 4096, 8192, 8193, 16383, 16384, 16385, 65535, 65536, 65537]
    )
    rows_list = [2, 4] if args.quick else [2, 3, 4, 8]
    mx.random.seed(7)
    longest = max(contexts) + max(rows_list)
    keys_all = mx.random.normal((1, kv_heads, longest, dim)).astype(mx.bfloat16)
    values_all = mx.random.normal((1, kv_heads, longest, dim)).astype(mx.bfloat16)
    mx.eval(keys_all, values_all)
    for rows in rows_list:
        for n in contexts:
            # The window's first row sees n keys, its last n + rows - 1.
            total = n + rows - 1
            keys, values = keys_all[:, :, :total], values_all[:, :, :total]
            queries = mx.random.normal((1, heads, rows, dim)).astype(mx.bfloat16)
            verify = split._row_exact_causal_sdpa(queries, keys, values, scale, limit)
            serial = mx.concatenate(
                [
                    mx.fast.scaled_dot_product_attention(
                        queries[:, :, r : r + 1],
                        keys[:, :, : n + r],
                        values[:, :, : n + r],
                        scale=scale,
                    )
                    for r in range(rows)
                ],
                axis=2,
            )
            plans = sorted({split._vector_plan(n + r, gqa, 1) for r in range(rows)})
            check(f"sdpa row-exact chunks R={rows} keys {n}..{total} plans={plans}", verify, serial)
            if not args.quick:
                # Informational: the plain chunking (no plan mirror) at the same window.
                shared = split._chunked_causal_sdpa(queries, keys, values, scale, limit)
                mx.eval(shared)
                if not same_bits(shared, serial):
                    report(
                        "INFO",
                        f"sdpa plain chunks (no plan mirror) R={rows} keys {n}..{total}",
                        f"differ, as expected at a plan boundary: {diff(shared, serial)}",
                    )

    # The QSA masked decode kernel (serial and verify rows past the 2048-token
    # budget) is a JIT copy of MLX's two-pass kernel with the mirrored
    # partition count; MLX's own call is the reference.
    for n in [1024, 8192, 16383, 16384, 65535, 65536]:
        q = mx.random.normal((1, heads, 1, dim)).astype(mx.bfloat16)
        k, v = keys_all[:, :, :n], values_all[:, :, :n]
        mask = mx.random.uniform(shape=(1, 1, 1, n)) < 0.3
        got = qsa_fast.masked_decode_sdpa(q, k, v, mask, scale)
        if got is None:
            report(
                "INFO",
                f"qsa masked decode keys={n}",
                "not engaged on this GPU class (serial and verify both run MLX)",
            )
            continue
        want = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        check(f"qsa masked decode (partitions {qsa_fast._two_pass_blocks(gqa, n)}) keys={n}", got, want)


# (name, K, N, bits, group size): the served Qwen3.8-Flash-Next oQ5e projections.
QMV_SHAPES = [
    ("attn.q_proj", 2560, 12288, 6, 64),
    ("attn.k_proj/v_proj", 2560, 512, 6, 64),
    ("attn.v_proj (gs128 layer)", 2560, 512, 6, 128),
    ("attn.o_proj", 6144, 2560, 6, 64),
    ("attn.index_qk_proj", 2560, 640, 6, 64),
    ("ple.key_proj", 2560, 10240, 6, 64),
    ("ple.value_proj", 2560, 2560, 6, 64),
    ("lm_head", 2560, 248320, 8, 64),
    ("gdn8.in_proj (stacked qkv|z|b|a)", 2560, 16480, 8, 64),
    ("gdn6.in_proj (stacked qkv|z|b|a)", 2560, 16480, 6, 64),
    ("gdn.out_proj", 6144, 2560, 5, 128),
    ("moe.shared_expert gate/up", 2560, 640, 8, 128),
    ("moe.shared_expert down", 640, 2560, 8, 128),
    ("moe.shared_expert_gate", 2560, 1, 8, 64),
    ("hc.down (fallback only)", 10240, 320, 6, 64),
    ("hc.up (fallback only)", 320, 10240, 6, 64),
]
# K = 256 (mod 512) with 6/8 bits: MLX 0.32.2 runs qmv_fast (K % 256), the
# transcription picks qmv (K % 512). No served Qwen4 projection has this K.
QMV_ALIGNMENT_PROBES = [
    ("alignment probe 8-bit K%512=256", 768, 1024, 8, 64),
    ("alignment probe 6-bit K%512=256", 1280, 1024, 6, 64),
]


@guarded
def qmv(args):
    from omlx.patches import row_exact_qmv

    rows_list = [2, 4] if args.quick else [2, 3, 4, 8]
    for index, (name, k, n, bits, gs) in enumerate(QMV_SHAPES + QMV_ALIGNMENT_PROBES):
        linear = quantized_linear(k, n, bits, gs, seed=100 + index)
        failed, jit_failed = [], []
        for rows in rows_list:
            x = (mx.random.normal((1, rows, k)) * 0.5).astype(mx.bfloat16)
            got = row_exact_qmv.quantized_linear(linear, x)
            want = stock_rows(linear, x)
            one_row = jit_rows(linear, x)
            mx.eval(got, want)
            if not same_bits(got, want):
                failed.append(f"R={rows}: {diff(got, want)}")
            if one_row is not None:
                mx.eval(one_row)
                if not same_bits(got, one_row):
                    jit_failed.append(f"R={rows}: {diff(got, one_row)}")
        if not name.startswith("alignment probe"):
            label = f"qmv {name} JIT one-row tile per row vs verify JIT"
            if jit_failed:
                report("FAIL", label, "; ".join(jit_failed))
            else:
                report("PASS", label)
        label = f"qmv {name} {k}->{n} {bits}-bit gs{gs} stock vs verify JIT"
        if failed and name.startswith("alignment probe"):
            # Device independent and not a served Qwen4 shape: reported, not failed.
            report("WARN", label, "latent transcription gap: " + "; ".join(failed))
        elif failed:
            report("FAIL", label, "; ".join(failed))
        else:
            report("PASS", label)
        # The serial one-row JIT tile (GDN decode plan) and every verify tile
        # geometry, where the layout runs qmv_fast.
        if k % 512 == 0 and n % 8 == 0 and n >= 16:
            bad = []
            x = (mx.random.normal((1, 1, k)) * 0.5).astype(mx.bfloat16)
            want = stock_rows(linear, x)
            for rps in (1, 2, 4):
                launch = row_exact_qmv.one_row_qmv(
                    linear.weight, linear.scales, linear.biases, bits, gs, "affine",
                    mx.bfloat16, rps,
                )
                if launch is None:
                    continue
                got = launch(x)
                mx.eval(got)
                if not same_bits(got, want):
                    bad.append(f"one_row rps={rps}: {diff(got, want)}")
            for rows in rows_list:
                xr = (mx.random.normal((1, rows, k)) * 0.5).astype(mx.bfloat16)
                want_r = stock_rows(linear, xr)
                for rps in (1, 2, 4):
                    for per_group in (d for d in (1, 2, 3, 4) if rows % d == 0):
                        launch = row_exact_qmv.rows_qmv(
                            linear.weight, linear.scales, linear.biases, bits, gs,
                            "affine", mx.bfloat16, lambda _, g=(rps, per_group): g,
                        )
                        if launch is None:
                            continue
                        got = launch(xr)
                        mx.eval(got)
                        if not same_bits(got, want_r):
                            bad.append(f"rows R={rows} rps={rps} rows/tg={per_group}: {diff(got, want_r)}")
            label = f"qmv {name} one-row + RowsQmv tile geometries vs stock"
            if bad:
                report("FAIL", label, "; ".join(bad[:4]) + (" ..." if len(bad) > 4 else ""))
            else:
                report("PASS", label)
        del linear
    # Verify launches q/k/v (and the GDN in-projections) as one grouped kernel.
    group = [quantized_linear(2560, n, 6, 64, seed=300 + i) for i, n in enumerate((12288, 512, 512))]
    for rows in rows_list:
        x = (mx.random.normal((1, rows, 2560)) * 0.5).astype(mx.bfloat16)
        outs = row_exact_qmv.quantized_linears(group, x)
        for name, linear, got in zip(("q", "k", "v"), group, outs):
            check(f"qmv attn grouped q/k/v launch ({name}) R={rows} vs stock", got, stock_rows(linear, x))
    # FP32 exposes accumulation differences a bf16 output rounds away.
    for name, k, n, bits, gs in [("attn.o_proj fp32", 6144, 2560, 6, 64), ("lm_head fp32 slice", 2560, 8192, 8, 64)]:
        mx.random.seed(11)
        w = mx.random.normal((n, k)) * 0.05
        linear = nn.QuantizedLinear(k, n, bias=False, group_size=gs, bits=bits)
        linear.weight, linear.scales, linear.biases = mx.quantize(w, group_size=gs, bits=bits)
        x = mx.random.normal((1, 4, k)) * 0.5
        check(f"qmv {name} (FP32 accumulators)", row_exact_qmv.quantized_linear(linear, x), stock_rows(linear, x))


def _gdn_module(signatures, seed):
    language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
    config = _text_config()
    hk, hv = config.linear_num_key_heads, config.linear_num_value_heads
    dk, dv = config.linear_key_head_dim, config.linear_value_head_dim
    conv = 2 * hk * dk + hv * dv
    mx.random.seed(seed)
    module = language.Qwen4ExpGatedDeltaNet(config)
    module.conv1d.weight = (mx.random.normal((conv, 4, 1)) * 0.3).astype(mx.bfloat16)
    module.norm.weight = (1 + mx.random.normal((dv,)) * 0.1).astype(mx.bfloat16)
    module.A_log = (mx.random.normal((hv,)) * 0.5).astype(mx.bfloat16)
    module.dt_bias = (mx.random.normal((hv,)) * 0.5).astype(mx.bfloat16)
    for name, n, (bits, gs) in zip(
        ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"), (conv, hv * dv, hv, hv), signatures
    ):
        setattr(module, name, quantized_linear(HIDDEN, n, bits, gs, seed=seed + n))
    module.out_proj = quantized_linear(hv * dv, HIDDEN, 5, 128, seed=seed + 1)
    module.eval()
    mx.eval(module.parameters())
    return module, conv, hv, dv, dk


@guarded
def gdn(args):
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.speculative.cache_state import start_speculative_cache

    from omlx.patches import qwen35_gdn_prework as prework

    q4 = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
    verifier_cls = getattr(q4, "_Qwen4Verifier", None) or q4.Qwen4ExpBatchInvariantForward
    rows_list = [2, 4] if args.quick else [2, 3, 4, 5, 8]
    layouts = [
        ("6-bit layer (serial: JIT decode plan)", ((6, 64),) * 4, False),
        ("8-bit layer (serial: stock mlx-vlm chain)", ((8, 64),) * 4, False),
        # qwen4_gdn_decode_wide_proj: the decode plan admits the 8-bit layer too.
        ("8-bit layer, wide decode plan (serial: JIT decode plan)", ((8, 64),) * 4, True),
    ]
    for label, signatures, wide in layouts:
        module, conv, hv, dv, dk = _gdn_module(signatures, seed=5)
        if wide:
            module._omlx_qwen4_wide_projections = True
        plan = prework._qwen4_decode_plan(module)
        report("INFO", f"gdn {label}", f"decode plan {'engaged' if plan is not None else 'None -> stock serial path'}")
        for rows in rows_list:
            mx.random.seed(40 + rows)
            inputs = (mx.random.normal((1, rows, HIDDEN)) * 0.5).astype(mx.bfloat16)
            conv_state = (mx.random.normal((1, 3, conv)) * 0.5).astype(mx.bfloat16)
            state = mx.random.normal((1, hv, dv, dk)) * 0.05
            serial_cache = ArraysCache(size=2)
            serial_cache[0], serial_cache[1] = conv_state, state
            outputs, states = [], []
            for t in range(rows):
                outputs.append(module(inputs[:, t : t + 1], cache=serial_cache))
                mx.eval(outputs[-1], serial_cache[0], serial_cache[1])
                states.append(serial_cache[1])
            cache = ArraysCache(size=2)
            cache[0], cache[1] = conv_state, state
            transaction = start_speculative_cache([cache], rows)
            launches = []
            step = prework.qwen4_verify_step_fused

            def counted(*a, _step=step):
                launches.append(1)
                return _step(*a)

            prework.qwen4_verify_step_fused = counted
            try:
                with armed():
                    got = verifier_cls()._gated_delta(module, inputs, None, cache)
                mx.eval(got, cache.state)
            finally:
                prework.qwen4_verify_step_fused = step
            path = "fused verify" if launches else "per-op verify"
            serial = mx.concatenate([o for o in outputs], axis=1)
            ok = check(f"gdn {label} R={rows} ({path}) output rows", got, serial)
            if ok:
                check(f"gdn {label} R={rows} final recurrent state", cache[1], states[-1])
            transaction.abort()
        del module


def _hc_module(bits: int, inject: bool, seed: int):
    language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
    hc, lowrank = 4, 320
    width = hc * HIDDEN
    mx.random.seed(seed)
    module = language.Qwen4ExpGatedResidual.__new__(language.Qwen4ExpGatedResidual)
    nn.Module.__init__(module)
    module.hc_count, module.hidden_size, module.hc_lowrank = hc, HIDDEN, lowrank
    module.hc_norm = language.Qwen4ExpRMSNorm(width, group_size=HIDDEN, eps=1e-6)
    module.hc_norm.weight = (mx.random.normal((width,)) * 0.05).astype(mx.bfloat16)
    projections = [("input_mix_weight_down", width, lowrank), ("input_mix_weight_up", lowrank, width)]
    if inject:
        projections.append(("block_inject_weight", width, hc))
    for name, k, n in projections:
        linear = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=bits)
        linear.scales = (mx.abs(mx.random.normal(linear.scales.shape)) * 0.01 + 0.002).astype(mx.bfloat16)
        linear.biases = (mx.random.normal(linear.biases.shape) * 0.005).astype(mx.bfloat16)
        setattr(module, name, linear)
    mx.eval(module.parameters())
    return module


def _flatten(out):
    return list(out) if isinstance(out, tuple) else [out]


@guarded
def hc(args):
    hc_fused = importlib.import_module("mlx_vlm.models.qwen4_exp.hc_fused")
    rows_list = [1, 2, 4, 7] if args.quick else [1, 2, 3, 4, 5, 6, 7, 8]
    width = 4 * HIDDEN
    for bits in (6, 8):
        for inject in (True, False):
            module = _hc_module(bits, inject, seed=bits * 10 + inject)
            kind = "block (inject)" if inject else "mixer"
            for use_write in (False, True):
                for rows in rows_list:
                    mx.random.seed(rows * 31 + bits)
                    x = mx.random.normal((1, rows, width)).astype(mx.bfloat16)
                    write = None
                    if use_write:
                        write = (
                            (mx.random.normal((1, rows, HIDDEN)) * 0.5).astype(mx.bfloat16),
                            (mx.random.uniform(shape=(1, rows, 4)) * 2).astype(mx.bfloat16),
                        )
                    fused_probe = hc_fused.fused_forward(module, x, write)
                    if fused_probe is None:
                        report("FAIL", f"hc {kind} {bits}-bit rows={rows} write={use_write}", "fused kernels fell back (see flags)")
                    with armed():
                        verify = _flatten(module(x, target_verify=True, write=write))
                    singles = []
                    for r in range(rows):
                        w = None if write is None else (write[0][:, r : r + 1], write[1][:, r : r + 1])
                        singles.append(_flatten(module(x[:, r : r + 1], write=w)))
                    serial = [mx.concatenate([s[i] for s in singles], axis=1) for i in range(len(singles[0]))]
                    label = f"hc {kind} {bits}-bit R={rows} write={use_write} ({'V2' if rows <= 6 else 'V1'} kernels)"
                    bad = [diff(g, s) for g, s in zip(verify, serial) if (mx.eval(g, s) or not same_bits(g, s))]
                    if len(verify) != len(serial) or bad:
                        report("FAIL", label, "; ".join(bad) or "output arity differs")
                    elif rows > 1 or not args.quick:
                        report("PASS", label)
            del module


@guarded
def moe(args):
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    from omlx.patches import qwen35_moe_routed_decode as routed
    from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

    class _FakeQwen4Model:
        pass

    _FakeQwen4Model.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"
    mx.random.seed(0)
    block_args = SimpleNamespace(
        hidden_size=HIDDEN,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        num_experts=512,
        num_experts_per_tok=10,
    )
    block = Qwen3_5MoeSparseMoeBlock(block_args)
    block.set_dtype(mx.bfloat16)
    sm = block.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(sm, name, getattr(sm, name).to_quantized(64, 5))
    shared = block.shared_expert
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(shared, name, nn.QuantizedLinear.from_linear(getattr(shared, name), 128, 8))
    block.shared_expert_gate = nn.QuantizedLinear.from_linear(block.shared_expert_gate, 64, 8)
    block.eval()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("mlp.switch_mlp", sm)]
    fused = apply_qwen35_moe_gate_up_fusion(model)
    mx.eval(block.parameters())
    report("INFO", "moe block", f"512 experts 5-bit gs64, shared 8-bit gs128; gate/up fused={fused}")
    engaged = []
    window = routed.routed_verify_window

    def spy(b, x, _window=window):
        y = _window(b, x)
        engaged.append(y is not None)
        return y

    routed.routed_verify_window = spy
    try:
        for rows in ([2, 4] if args.quick else [2, 3, 4, 8]):
            for style in ("independent", "consecutive"):
                mx.random.seed(rows * 7 + len(style))
                if style == "independent":
                    x = (mx.random.normal((1, rows, HIDDEN)) * 1.5).astype(mx.bfloat16)
                else:
                    base = mx.random.normal((1, 1, HIDDEN))
                    x = (0.97 * base + 0.25 * mx.random.normal((1, rows, HIDDEN))).astype(mx.bfloat16)
                serial = mx.concatenate(
                    [block(x[:, r : r + 1].reshape(1, 1, 1, HIDDEN)).reshape(1, 1, HIDDEN) for r in range(rows)],
                    axis=1,
                )
                count = len(engaged)
                with armed():
                    got = Qwen3_5BatchInvariantForward()._feed_forward(block, x)
                mx.eval(got, serial)
                path = "fused window" if engaged[count:] == [True] else "composed verifier"
                check(f"moe R={rows} {style} rows ({path}) vs fused one-token decode", got, serial)
    finally:
        routed.routed_verify_window = window


def _text_config():
    """Qwen3.8-Flash-Next layer dimensions (few layers/experts: only single
    modules are built from it)."""
    qwen4_exp = importlib.import_module("mlx_vlm.models.qwen4_exp")
    return qwen4_exp.TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        output_gate_type="sigmoid",
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=262144,
        hc_count=2,
        hc_lowrank=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        ple_layer_ids=[],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        eos_token_id=1,
        rope_parameters={
            "type": "default",
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
        },
    )


def _attention_setup():
    language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
    mx.random.seed(8)
    attn = language.Qwen4ExpAttention(_text_config())
    for _, module in attn.named_modules():
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            module.weight = (mx.random.normal(module.weight.shape) * fan_in**-0.5).astype(mx.bfloat16)
    for norm in (attn.q_norm, attn.k_norm, attn.indexer.q_layernorm, attn.indexer.k_layernorm):
        norm.weight = mx.random.normal(norm.weight.shape) * 0.1
    nn.quantize(attn, group_size=64, bits=6, class_predicate=lambda _, m: isinstance(m, nn.Linear))
    for _, module in attn.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.scales = module.scales.astype(mx.bfloat16)
            module.biases = module.biases.astype(mx.bfloat16)
    mx.eval(attn.parameters())
    return language, attn


def _positions(start: int, length: int) -> mx.array:
    seq = mx.arange(start, start + length, dtype=mx.int32)[None]
    return mx.broadcast_to(seq[None], (3, 1, length))


@guarded
def attn(args):
    language, module = _attention_setup()

    def state(cache):
        arrays = [
            cache.keys[..., : cache.offset, :],
            cache.values[..., : cache.offset, :],
            cache.index_keys,
            cache.index_position_ids,
        ]
        copies = [mx.array(a) for a in arrays if a is not None]
        mx.eval(copies)
        return copies

    contexts = [700, 1022, 2060] if args.quick else [700, 1020, 1022, 1500, 2060, 4000]
    for context in contexts:
        cache = language.QSAKVCache()
        mx.random.seed(9)
        done = 0
        while done < context:
            size = min(4096, context - done)
            x = mx.random.normal((1, size, HIDDEN)).astype(mx.bfloat16)
            module(x, mask="causal", cache=cache, position_ids=_positions(done, size))
            mx.eval([a for a in cache.state if a is not None])
            done += size
        for rows in ([2, 4] if args.quick else [2, 3, 4, 8]):
            mx.random.seed(31 + rows)
            x = (mx.random.normal((1, rows, HIDDEN)) * 0.5).astype(mx.bfloat16)
            with armed():
                verified = module(
                    x, mask="causal", cache=cache,
                    position_ids=_positions(cache.offset, rows), target_verify=True,
                )
                mx.eval(verified)
            verified_state = state(cache)
            cache.trim(rows)
            outs = []
            for r in range(rows):
                outs.append(module(x[:, r : r + 1], mask=None, cache=cache, position_ids=_positions(cache.offset, 1)))
                mx.eval(outs[-1])
            serial = mx.concatenate(outs, axis=1)
            serial_state = state(cache)
            cache.trim(rows)
            arm = "dense" if context + rows <= 2048 else "past budget"
            ok = check(f"attn {arm} context={context} R={rows} output rows", verified, serial)
            if ok:
                bad = [diff(g, w) for g, w in zip(verified_state, serial_state) if not same_bits(g, w)]
                if bad:
                    report("FAIL", f"attn {arm} context={context} R={rows} cache state", "; ".join(bad))


@guarded
def ple(args):
    language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
    channels, taps, dilation = 4 * HIDDEN, 4, 3
    conv = nn.Conv1d(channels, channels, taps, dilation=dilation, groups=channels, bias=False)
    mx.random.seed(3)
    conv.weight = (mx.random.normal((channels, taps, 1)) * 0.3).astype(mx.bfloat16)
    state_len = (taps - 1) * dilation
    prefill = mx.random.normal((1, 64 + state_len, channels)).astype(mx.bfloat16)
    language._depthwise_conv1d(conv, prefill)  # first use validates against mx.conv1d
    status = language._DEPTHWISE_CONV_STATE
    report(
        "INFO" if status["enabled"] else "FAIL",
        "ple depthwise conv kernel",
        "validated and enabled" if status["enabled"] else "disabled by first-use validation (see warnings)",
    )
    for rows in (2, 4):
        window = mx.random.normal((1, state_len + rows, channels)).astype(mx.bfloat16)
        got = language._depthwise_conv1d(conv, window)
        want = mx.concatenate(
            [language._depthwise_conv1d(conv, window[:, r : r + state_len + 1]) for r in range(rows)], axis=1
        )
        check(f"ple conv (served path) R={rows} vs one-row calls", got, want)
        got = conv(window)
        want = mx.concatenate([conv(window[:, r : r + state_len + 1]) for r in range(rows)], axis=1)
        mx.eval(got, want)
        if same_bits(got, want):
            report("INFO", f"ple conv fallback mx.conv1d R={rows} row-exact")
        else:
            report("INFO", f"ple conv fallback mx.conv1d R={rows} NOT row-exact", diff(got, want))


@guarded
def flags(args):
    language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
    hc_fused = importlib.import_module("mlx_vlm.models.qwen4_exp.hc_fused")
    qsa_fast = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")
    from omlx.patches import qwen35_moe_routed_decode as routed

    state = {
        "hc_fused failure logged": hc_fused._FAILURE_LOGGED,
        "hc NAX prefill broken": hc_fused._NAX_BROKEN,
        "moe routed decode disabled": routed._DISABLED,
        "moe verify window disabled": routed._WINDOW_DISABLED,
        "qsa masked decode sdpa disabled": qsa_fast._DECODE_SDPA_DISABLED,
        "ple conv kernel enabled": language._DEPTHWISE_CONV_STATE["enabled"],
    }
    bad = {k: v for k, v in state.items() if v != (k == "ple conv kernel enabled")}
    report("FAIL" if bad else "PASS", "fail-closed flags", ", ".join(f"{k}={v}" for k, v in state.items()))
    relevant = [w for w in WARNINGS if any(s in w for s in ("fail", "disabled", "differ", "fallback"))]
    for line in relevant[:20]:
        report("INFO", "warning", line)


SECTIONS = {
    "env": env,
    "sdpa": sdpa,
    "qmv": qmv,
    "gdn": gdn,
    "hc": hc,
    "moe": moe,
    "attn": attn,
    "ple": ple,
    "flags": flags,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="fewer rows and contexts")
    parser.add_argument("--only", default="", help="comma-separated sections")
    parser.add_argument("--list", action="store_true", help="list sections and exit")
    args = parser.parse_args()
    if args.list:
        print("\n".join(SECTIONS))
        return 0
    if not mx.metal.is_available():
        print("Metal is not available: this diagnostic needs an Apple GPU.")
        return 2
    logging.getLogger().addHandler(_Capture(level=logging.WARNING))
    wanted = [s for s in args.only.split(",") if s] or list(SECTIONS)
    unknown = [s for s in wanted if s not in SECTIONS]
    if unknown:
        parser.error(f"unknown sections: {unknown}; see --list")
    if "env" not in wanted:
        wanted.insert(0, "env")
    if "flags" not in wanted:
        wanted.append("flags")
    patches()
    started = time.perf_counter()
    for name in wanted:
        print(f"\n== {name} ==", flush=True)
        SECTIONS[name](args)
    counts = {s: sum(1 for r in RESULTS if r[0] == s) for s in ("PASS", "FAIL", "ERROR", "WARN", "INFO")}
    print(
        f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['ERROR']} errors, "
        f"{counts['WARN']} warnings "
        f"in {time.perf_counter() - started:.0f}s on {mx.device_info().get('architecture')}"
    )
    for status, name, detail in RESULTS:
        if status in ("FAIL", "ERROR", "WARN"):
            print(f"  {status}: {name}  {detail}")
    return 1 if counts["FAIL"] or counts["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())
