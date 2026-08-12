# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tile sweep + validation for `gmm_v2`'s `transpose_rhs` path.

WHY THIS EXISTS. On a 512-device v7x run the `transpose_rhs=True` (dlhs) call measured
**2687 us/call at 27% of MXU peak and 23% of HBM peak with 0% DMA stall**, while the
`transpose_rhs=False` (forward) call on identical dims measured **1402 us at 52% MXU / 45% HBM**.
Same FLOPs, same operational intensity. Bound by neither ceiling => the transposed path was
losing time on-chip, not to memory or the MXU.

HYPOTHESIS (now CONFIRMED, see below). A transposed operand cannot be fed to the MXU directly; it
goes through the XLU transpose unit. In `inner_kernel` the transposed slice
`tiled_rhs[start_n:end_n, start_k:end_k]` is taken INSIDE the `start_n` x `b_id` loops, so the XLU
pass is repeated per sub-tile and amortized over only `tile_m` rows of lhs.
`pallas_mosaic_tpu_v2_tgmm_kernel.calculate_tgmm_tiling` documents the same effect for its own
(transposed) lhs and answers it the same way -- `bf16_bf16_tile_m = 256` where `calculate_tiling`
uses 128.

ROUND 1 RESULT (TPU7x, jax 0.8.3, m=65536, g=128 balanced, bf16, median of 20):

    k=3072 -> n=2048        us/call  TFLOP/s  %MXU
    transpose, tile_m=128      2684      307    27%
    transpose, tile_m=256      1133      728    63%
    transpose, tile_m=512      1066      774    67%
    no transpose               1065      774    67%   <- at tile_m=512 the transpose is FREE

so `calculate_tiling` now grows `tile_m` when `rhs_cfgs.transposed` is set.

WHAT ROUND 2 (this file) ADDS, i.e. what still has to be true for that to be a real win:

 1. **AUTO rows.** Report the tiles `calculate_tiling` actually returns, for transposed and not.
    This is the only way to see what production runs (and what VMEM capacity the part reports), and
    it is what proves the new policy fires instead of silently no-oping.
 2. **Ragged groups.** `fill_metadata` emits `ceil((group_size + off) / tile_m)` gm tiles PER GROUP,
    so the last gm tile of every group is partially filled and a bigger `tile_m` pads more. Round 1
    used a perfectly balanced router (group_size == 512 == tile_m), which is the best case and hides
    this entirely. Two adversarial distributions are added:
      * `skew50`   -- alternating 768 / 256 rows (a +-50% imbalanced router).
      * `padworst` -- alternating 513 / 511 rows. Pathological for tile_m=512: a 513-row group needs
                      two 512-row gm tiles, i.e. ~2x padding, while tile_m=128 pads almost nothing.
    If the win survives `padworst`, it survives any router.
 3. **Correctness.** `tile_m` must not change the maths. With `tile_k` unsplit, every output element
    accumulates in the same order regardless of `tile_m`, so tile_m=128 and tile_m=512 must agree
    **bit for bit** -- a much sharper test than a tolerance. The transposed and untransposed paths
    are also compared against each other (same maths, different addressing) and against an f32
    per-group einsum reference.

ROUND 2 RESULT: confirmed on hardware, but it CHANGED THE ANSWER -- tile_m=512 is 4% faster than 256
on a perfectly balanced router and 25-40% SLOWER on a ragged one, because `fill_metadata` emits
ceil((group_size + off) / tile_m) gm tiles PER GROUP and the padding grows with tile_m. 256 adopted.
Validated on 512 v7x devices: dlhs 264.6 -> 123.1 ms/step/chip, GMM total -19.4%, loss
bit-identical, forward and tgmm buckets unmoved.

WHAT ROUND 3 (this version) ADDS: the same lever for the UNTRANSPOSED path. Round 2 measured
untransposed tile_m=256 at 1121-1360 us vs 128's 1345-1547 across all three routers -- a ~17% win on
the other 16 of the 20 GMM calls per step, twice the size of what shipped. It was deliberately NOT
shipped with the transposed fix, because raising tile_m for `transposed=False` changes tiles for
EVERY gmm_v2 caller, and two production directions are not enough evidence for that. So the sweep is
now shape-driven (`SHAPES`) and covers tokamax's own perf-test shape, an n-at-the-floor shape, and
two shapes sitting on the mean-rows-per-group guard, each under all three routers. The decision rule
is in the summary footer: a tile_m that wins in EVERY (shape, router) row can be the default; one
that loses anywhere cannot, because `calculate_tiling` cannot see the router.

Usage (needs a TPU; one chip is enough -- weights are only ~1.6 GB):
    python -m tokamax._src.ops.ragged_dot.gmm_v2_kernel_tests.transpose_rhs_tile_sweep

MULTI-HOST NOTE. The benchmark needs exactly ONE chip, but the smallest schedulable v7x allocation
is a whole subblock -- Bastion rejects `7x-8`/`7x-16`/... with "Topology '7x-8' has 1 VMs per slice,
but requires at least 16 VMs per slice to support subblock size for 7x". So in practice this runs as
a MANY-process job (e.g. tpu-7x-256 = 32 VMs x 4 chips). Every process runs the identical sweep,
each on its OWN local device 0, and the arrays are `device_put` there so nothing is ever sharded
across the slice. There are no collectives, so the processes never interact -- the extra ranks are
just free repeats. Only process 0 prints.
"""

import sys
import time

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_gmm_kernel as gmm_lib
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_lib

# Shapes to sweep, as (label, size_m, num_groups, size_k, size_n). size_k/size_n are what the
# KERNEL sees, i.e. contract size_k -> emit size_n.
#
# The first two came off a 512-device v7x trace's kernel names (
# gmm_v2-g_128-m_65536-k_{2048,3072}-...-n_{3072,2048}-...). The rest exist because raising tile_m
# for the UNTRANSPOSED path would change tiles for EVERY gmm_v2 caller, not just the dlhs one --
# so the two production directions are not enough evidence on their own:
#   tkm_default -- tokamax's own perf-test shape (much bigger m and k, narrower n)
#   narrow_n    -- n at the tile_n floor, where a bigger tile_m buys the least
#   small_m     -- mean rows per group == 128, i.e. exactly at the growth guard
#   many_g      -- 512 groups over the same m: same mean, but 4x the per-group padding events
SHAPES = (
    ("prod_wi   ",  65536, 128, 3072, 2048),
    ("prod_wo   ",  65536, 128, 2048, 3072),
    ("tkm_defaul", 262144, 256, 7168, 1024),
    ("narrow_n  ",  65536, 128, 3072,  512),
    ("small_m   ",  16384, 128, 3072, 2048),
    ("many_g    ",  65536, 512, 3072, 2048),
)
# Kept for the correctness checks and the legacy 2-direction summary.
NUM_GROUPS = 128
SIZE_M = 65536
DIRECTIONS = ((2048, 3072), (3072, 2048))
# 192 and 1024 added for round 7 (fp8). Their jobs:
#   * 192 is the MECHANISM DISCRIMINATOR. Rounds 1-6 only sampled powers of two, so "a real 256
#     granularity" and "some unrelated power-of-two effect" are indistinguishable in that data.
#     192 is a legal tile_m (multiple of size_lhs_sublane=16) and is not a power of two, so:
#       192 == 256 (and 128 slower)  => a granularity exists at 256
#       192 between 128 and 256      => smooth, no granularity -- the step was a sampling artifact
#   * 1024 only fits under fp8 (bf16 needs 64.0 MiB of a 57.6 MiB budget at full tile_n; fp8 halves
#     the rhs term and drops the accumulator to 2 bytes, so it fits at 36.0 MiB). It tests whether
#     the optimum MOVES UP once VMEM stops binding. Under bf16 it is expected to raise; the runner
#     records the exception per-cell rather than aborting, so those rows are informative, not lost.
TILE_MS = (128, 192, 256, 512, 1024)
# See the module docstring. "balanced" is the round-1 case and the best case for a large tile_m.
GROUP_DISTS = ("balanced", "skew50", "padworst")
REPEATS = 20

# DTYPE AXIS (round 7). `transpose_rhs` + fp8 is admissible TODAY as long as no scale is passed:
# the sub-8-bit refusal tests `itemsize_bits(rhs.dtype) < 8` and fp8 is exactly 8, and the
# `rhs_scale`/`lhs_scale` refusals only fire when a scale array is present. A per-TENSOR fp8 recipe
# factors its scale out of the matmul entirely -- (a_s*A)@(b_s*B) = a_s*b_s*(A@B) -- so the kernel
# sees bare fp8 operands and the epilogue applies the scale outside. That is how a real fp8 MoE
# dlhs call already works, and it is why this sweep can cover the transposed path at fp8 without
# first implementing per-BLOCK `rhs_scale` + `transpose_rhs`.
#
# Why each row matters: `calculate_tiling` derives tile_m from the dtypes
# (`128 * lhs_mod // rhs_mod`, each mod in {1,2}), so the base is NOT dtype-invariant:
#   bf16 x bf16 -> base 128 -> our growth gives 256   (what shipped, 18/18 cells measured)
#   fp8  x fp8  -> base 128 -> our growth gives 256   (same number, UNMEASURED)
#   bf16 x fp8  -> base  64 -> our growth gives 128   (HALF -- i.e. the value we proved wrong)
# So which row applies to a given fp8 recipe depends on whether it quantizes both operands or only
# the weight. Read the consumer's quantization config before reading these numbers.
F8 = jnp.float8_e4m3fn
DTYPES = (
    (jnp.bfloat16, jnp.bfloat16, "bf16"),  # control: reproduces rounds 1-6, catches harness drift
    (F8, F8, "f8f8"),  # both operands quantized
    (jnp.bfloat16, F8, "bff8"),  # weight-only quantization -- the base-64 case
)


def group_sizes(kind: str, size_m: int = SIZE_M, num_groups: int = NUM_GROUPS) -> jax.Array:
  """Router distributions, all summing EXACTLY to `size_m` (drop-free MoE)."""
  per = size_m // num_groups
  if kind == "balanced":
    sizes = [per] * num_groups
  elif kind == "skew50":
    sizes = [per + per // 2 if i % 2 else per - per // 2 for i in range(num_groups)]
  elif kind == "padworst":
    # One row over / one row under a tile_m=512 boundary: worst case for gm-tile padding.
    sizes = [per + 1 if i % 2 else per - 1 for i in range(num_groups)]
  else:
    raise ValueError(f"unknown group dist {kind!r}")
  assert sum(sizes) == size_m, (kind, sum(sizes), size_m)
  return jnp.asarray(sizes, dtype=jnp.int32)


def auto_tiles(size_k: int, size_n: int, transposed: bool, vmem_limit: int,
               size_m: int = SIZE_M, num_groups: int = NUM_GROUPS,
               lhs_dt=jnp.bfloat16, rhs_dt=jnp.bfloat16):
  """What `calculate_tiling` returns for a production-shaped call at these dtypes.

  Mirrors the `Dimensions`/`InputConfigs` that `make_gmm_configs` builds for an unquantized
  gmm (no rhs_scale / rhs_bias / lhs_scale / fuse_act -- the only combination `transpose_rhs`
  supports), so this reports the tiles the real call site gets. `lhs_dt`/`rhs_dt` are threaded
  through because tile_m is dtype-derived (see DTYPES): the same shape gets a different tile_m
  under fp8 than under bf16, and that is one of the things this sweep is here to expose.
  """
  info = pltpu.get_tpu_info()
  dims = gmm_lib.Dimensions(
      size_m=size_m, size_k=size_k, size_n=size_n,
      size_group=num_groups, size_lhs_group=num_groups,
      size_lhs_sublane=min(info.get_sublane_tiling(jnp.dtype(lhs_dt)), size_m),
  )
  lhs_cfgs = gmm_lib.InputConfigs(
      quant_dtype=None, quant_block_size=512, dtype=jnp.dtype(lhs_dt)
  )
  rhs_cfgs = gmm_lib.InputConfigs(
      quant_dtype=None, quant_block_size=size_k, dtype=jnp.dtype(rhs_dt),
      transposed=transposed,
  )
  return gmm_lib.calculate_tiling(dims, lhs_cfgs, rhs_cfgs, vmem_limit)


def _time_us(fn, *args, repeats: int = REPEATS) -> float:
  """Median wall time per call, in us. Warm up first so compile is excluded."""
  jax.block_until_ready(fn(*args))
  ts = []
  for _ in range(repeats):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    ts.append((time.perf_counter() - t0) * 1e6)
  ts.sort()
  return ts[len(ts) // 2]


def _make_fn(tiles, transpose: bool):
  """jit'd gmm_v2. `tiles=None` means: use the default `calculate_tiling` (the AUTO row)."""
  kwargs = {} if tiles is None else {"tile_info": tiles}
  return jax.jit(
      lambda l, r, g: gmm_lib.gmm_v2(
          lhs=l, rhs=r, group_sizes=g,
          preferred_element_type=jnp.bfloat16,
          maybe_quantize_lhs=False,
          transpose_rhs=transpose,
          **kwargs,
      )
  )


def _fmt_row(row: tuple) -> str:
  label, size_k, size_n, dist, transpose, tile_m, tile_n, us, tf, note = row
  us_s = "SKIP" if us != us else f"{us:9.0f}"      # NaN check
  tf_s = "  -" if tf != tf else f"{tf:8.0f}"
  return (f"{label:<16}{f'{size_k}->{size_n}':<12}{dist:<10}{str(transpose):<7}"
          f"{str(tile_m):>8}{str(tile_n):>12}{us_s:>10}{tf_s}  {note}")


def make_inputs(size_k: int, size_n: int, device=None,
                size_m: int = SIZE_M, num_groups: int = NUM_GROUPS,
                lhs_dt=jnp.bfloat16, rhs_dt=jnp.bfloat16):
  """lhs + both rhs layouts, committed to one local device.

  `jax.random.normal` has no fp8 path, so fp8 operands are generated in bf16 and cast. Values are
  irrelevant to what this measures (TPU matmul timing is data-independent); only the dtype, and
  hence the byte width the tiling math sees, matters.
  """
  k0, k1 = jax.random.split(jax.random.PRNGKey(0))
  lhs = jax.random.normal(k0, (size_m, size_k), jnp.bfloat16).astype(lhs_dt)
  # transpose_rhs=False wants rhs [G, size_k, size_n];
  # transpose_rhs=True  wants the SAME axes swapped: [G, size_n, size_k].
  # (Caller contract: with the flag set you pass the weight UN-swapped relative to its own
  # forward layout -- see the kernel's `transpose_rhs` docstring. Both produce [M, size_n].)
  rhs_f = jax.random.normal(k1, (num_groups, size_k, size_n), jnp.bfloat16).astype(rhs_dt)
  rhs_t = jnp.swapaxes(rhs_f, 1, 2)  # [G, size_n, size_k]
  # COMMIT every input to one local device. On a multi-process slice this is what keeps the
  # computation single-device: committed inputs pin the jit, so no input is ever resharded and no
  # collective is introduced.
  if device is not None:
    lhs, rhs_f, rhs_t = (jax.device_put(x, device) for x in (lhs, rhs_f, rhs_t))
  return lhs, rhs_f, rhs_t


def run_shape(shape, vmem_limit, device=None, report=None,
              dtypes=(jnp.bfloat16, jnp.bfloat16, "bf16")) -> list[tuple]:
  """Sweep tile_m x transpose_rhs x group-distribution for ONE (m, g, k, n) shape and dtype pair.

  The dtype tag is folded into the shape label (`prod_wi/f8f8`) rather than added as a new row
  field, so `_fmt_row` and both summary groupers keep working unchanged on a 10-tuple.
  """
  label, size_m, num_groups, size_k, size_n = shape
  lhs_dt, rhs_dt, dt_tag = dtypes
  label = f"{label.strip()}/{dt_tag}"
  lhs, rhs_f, rhs_t = make_inputs(size_k, size_n, device, size_m, num_groups, lhs_dt, rhs_dt)
  gs = {d: group_sizes(d, size_m, num_groups) for d in GROUP_DISTS}
  if device is not None:
    gs = {d: jax.device_put(v, device) for d, v in gs.items()}
  flops = 2 * size_m * size_k * size_n

  # tile_n is always full width in the forced rows; which tile_n is admissible is
  # `calculate_tiling`'s job, and the AUTO rows report what it picked.
  configs = [(None, tr) for tr in (False, True)]  # AUTO first: it is the row that matters
  configs += [
      (gmm_lib.TileSizes(tile_m=tm, tile_k=size_k // kdiv, tile_n=size_n), tr)
      for tr in (False, True)
      for tm in TILE_MS
      for kdiv in GMM_TILE_K_DIVS
      if tm <= size_m and (size_k // kdiv) >= pltpu.get_tpu_info().num_lanes
  ]

  out = []
  for tiles, transpose in configs:
    if tiles is None:
      try:
        chosen = auto_tiles(size_k, size_n, transpose, vmem_limit, size_m, num_groups,
                            lhs_dt, rhs_dt)
        tm_s, tn_s = f"AUTO:{chosen.tile_m}", f"AUTO:{chosen.tile_n}"
      except Exception as e:  # noqa: BLE001
        row = (label, size_k, size_n, "-", transpose, "AUTO:?", "AUTO:?", float("nan"),
               float("nan"), f"calculate_tiling: {type(e).__name__}: {e}"[:90])
        out.append(row)
        if report is not None:
          report(row)
        continue
    else:
      tm_s = str(tiles.tile_m)
      tn_s = str(tiles.tile_n) + ("" if tiles.tile_k == size_k else f"/k{tiles.tile_k}")
    fn = _make_fn(tiles, transpose)
    for dist in GROUP_DISTS:
      key = (label, size_k, size_n, dist, transpose, tm_s, tn_s)
      try:
        us = _time_us(fn, lhs, rhs_t if transpose else rhs_f, gs[dist])
        out.append(key + (us, flops / (us * 1e-6) / 1e12, ""))
      except Exception as e:  # noqa: BLE001 - a non-admissible tile must not kill the sweep
        msg = type(e).__name__ + ": " + str(e).replace("\n", " ")[:70]
        out.append(key + (float("nan"), float("nan"), msg))
      # Report as we go, not at the end: a preemption or a hang in a later config must not cost
      # us the rows already measured.
      if report is not None:
        report(out[-1])
  return out


def check_correctness(size_k, size_n, device=None) -> list[str]:
  """Does the tiling change (or the transpose) alter the result?

  Two scales, because the two questions have different cheapest tests:

  A. AT PRODUCTION SIZE, no reference needed -- and this is the sharp test for the tiling change.
     `tile_m=512` vs `tile_m=128` on the SAME path must be **BIT-IDENTICAL**: `tile_k` is unsplit in
     both, so every output element accumulates over the full k in one pass and in the same order.
     Any difference at all means tile_m is changing the maths rather than the schedule. The
     transposed and untransposed paths are also differenced against each other (same maths,
     different addressing; the accumulation order over sub-tiles of n/k can differ, so this one is
     a magnitude check, not a bit check).

  B. AT A SMALL SIZE, against `jax.lax.ragged_dot` in f32 -- catches an answer that is
     self-consistently wrong. Deliberately NOT done at production size: an f32 reference there
     means a 3.2 GB f32 rhs and ~0.8 TFLOP of reference matmul, which is pure cost for no extra
     signal.
  """
  msgs = []

  # --- A. bit-identity at production size --------------------------------------------------
  lhs, rhs_f, rhs_t = make_inputs(size_k, size_n, device)
  gs = group_sizes("skew50")
  if device is not None:
    gs = jax.device_put(gs, device)

  def run(tile_m: int, transpose: bool):
    tiles = gmm_lib.TileSizes(tile_m=tile_m, tile_k=size_k, tile_n=size_n)
    return _make_fn(tiles, transpose)(lhs, rhs_t if transpose else rhs_f, gs)

  for transpose in (False, True):
    try:
      a, b = run(128, transpose), run(512, transpose)
      same = bool(jnp.array_equal(a, b))
      msg = (f"  [prod {size_k}->{size_n}] transpose_rhs={transpose!s:5} tile_m 128 vs 512: "
             f"{'BIT-IDENTICAL' if same else 'DIFFERS'}")
      if not same:
        d = jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))
        msg += f" max|d|={float(jnp.max(d)):.3e} mean|d|={float(jnp.mean(d)):.3e}"
      msgs.append(msg)
    except Exception as e:  # noqa: BLE001
      msgs.append(f"  [prod] transpose_rhs={transpose} FAILED: {type(e).__name__}: {str(e)[:110]}")

  try:
    untransposed = run(512, False).astype(jnp.float32)
    d = jnp.abs(untransposed - run(512, True).astype(jnp.float32))
    msgs.append(f"  [prod {size_k}->{size_n}] transposed vs untransposed (tile_m=512): "
                f"max|d|={float(jnp.max(d)):.3e} (max|out|={float(jnp.max(jnp.abs(untransposed))):.3e})")
  except Exception as e:  # noqa: BLE001
    msgs.append(f"  [prod] transposed-vs-untransposed FAILED: {type(e).__name__}: {str(e)[:110]}")

  # --- B. f32 ragged_dot reference at a small size ------------------------------------------
  sm, sk, sn, sg = 4096, 512, 768, 8
  try:
    k0, k1 = jax.random.split(jax.random.PRNGKey(1))
    l = jax.random.normal(k0, (sm, sk), jnp.bfloat16)
    rf = jax.random.normal(k1, (sg, sk, sn), jnp.bfloat16)
    rt = jnp.swapaxes(rf, 1, 2)
    # Ragged, non-uniform, summing to sm.
    sizes = [sm // sg + (64 if i % 2 else -64) for i in range(sg)]
    g = jnp.asarray(sizes, jnp.int32)
    if device is not None:
      l, rf, rt, g = (jax.device_put(x, device) for x in (l, rf, rt, g))
    ref = jax.jit(lambda a, b, c: jax.lax.ragged_dot(
        a.astype(jnp.float32), b.astype(jnp.float32), c))(l, rf, g)
    ref_mag = float(jnp.max(jnp.abs(ref)))
    for transpose in (False, True):
      for tile_m in (128, 512):
        tiles = gmm_lib.TileSizes(tile_m=tile_m, tile_k=sk, tile_n=sn)
        got = _make_fn(tiles, transpose)(l, rt if transpose else rf, g)
        d = jnp.abs(got.astype(jnp.float32) - ref)
        msgs.append(f"  [small m={sm} g={sg} {sk}->{sn}] transpose_rhs={transpose!s:5} "
                    f"tile_m={tile_m:<4} vs f32 ragged_dot: max|d|={float(jnp.max(d)):.3e} "
                    f"rel={float(jnp.max(d)) / (ref_mag + 1e-30):.2e}")
  except Exception as e:  # noqa: BLE001
    msgs.append(f"  [small] reference check FAILED: {type(e).__name__}: {str(e)[:140]}")
  return msgs


# tile_m values to try for TGMM. NOTE tgmm's `tile_m` is NOT gmm's: in tgmm it blocks the
# CONTRACTED dimension (out[g] = lhs[rows of g].T @ dout[rows of g], contracting over m rows), so it
# controls how many rows are accumulated per pass and how big each XLU transpose of the lhs tile is.
TGMM_TILE_MS = (128, 256, 512, 1024)
# ROUND 2 (r5). r4 answered the first question and raised a better one. At FULL-WIDTH tile_n, the
# hardcoded tile_m=256 is already the best admissible tile -- AUTO == 256 to within noise, 128 is
# 1.7x WORSE (which also falsifies the kernel's own comment that "any size less than 256 will have
# the same perf as using 256"), and 512/1024 do not fit VMEM at all:
#
#     k=3072 -> n=2048, us/call      balanced   skew50   padworst
#     AUTO (= 256)                       1315     1275       1538
#     tile_m=128                         2233     2229       2467
#     tile_m=256                         1306     1275       1548
#     tile_m=512 / 1024                  OOM      OOM        OOM   (vmem)
#
# So the binding constraint is VMEM, and the reason is visible in `within_vmem_limit`: the
# accumulator plus double-buffered output cost `tile_k * tile_n * (acc_bytes + 2*out_bytes)` at FULL
# k x n -- for k=3072, n=2048 that is 3072*2048*(4+4) = 50.3 MB of a 57.6 MB budget, leaving no room
# to deepen tile_m. Narrowing tile_n frees it: at n/2 that term halves to 25 MB.
#
# That combination -- deeper tile_m bought with a narrower tile_n -- is exactly the trade that
# mattered on the gmm side, and it has never been measured for tgmm. Hence the (tile_m, tile_n) grid
# below rather than tile_m alone.
TGMM_TILE_N_DIVS = (1, 2, 4)
# CLOSED by r6, so round 7 does not re-sweep it. r6 measured the split at prod_wi across
# 3 tile_m x 3 routers: split lost all 9 paired comparisons, by +60% (tile_m=128), +17% (256) and
# +6% (512). Two reasons not to spend fp8 machine time on it again:
#   * MECHANISM SAYS fp8 MAKES IT WORSE, not better. The cost of a split is the 24 MiB accumulator
#     round-tripping through VMEM, and that is unchanged under fp8 (with bare fp8 operands and no
#     scale, `quant_dtype` stays None, so `acc_dtype` stays f32). Meanwhile the MXU work per k-step
#     HALVES. Same fixed cost amortised over less compute => a larger relative penalty.
#   * IT IS NOT REACHABLE ANYWAY. `calculate_tiling` returns num_k=1 at every shape we run, and its
#     tile_k shrink branch is dead code for all but 3 of 512 size_n values: the `while` halts when
#     tile_n == tile_n_limit but the `if` requires tile_n < tile_n_limit. Sweeping it only
#     re-validates a guard.
# Restore `(1, 2)` if the guard's premise ever needs re-checking on new hardware.
GMM_TILE_K_DIVS = (1,)


def run_tgmm_shape(shape, device=None, report=None) -> list[tuple]:
  """Sweep tile_m x group-distribution for ONE shape of the drhs (weight-gradient) kernel.

  WHY THIS EXISTS. After the gmm_v2 tiling fix, `tgmm_v2` is the *slowest-served* of the three expert
  GMMs in the production trace: lowest measured HBM bandwidth (372 GiB/s vs gmm's 588) and the
  highest per-instance cost, at ~25% of all expert-GMM time. And unlike gmm, its tiling has never
  been measured -- only reasoned about. `calculate_tgmm_tiling` hardcodes:

      bf16_bf16_tile_m = 256
      # "because the mxu size is 256 ... any size less than 256 will have the same perf as using
      #  256" ... "Since we use it in MOE, the m can be dynamic and small. So we don't want it to be
      #  too big."

  Read that carefully: it argues 256 is a FLOOR (below it nothing changes) and declines to go higher
  on a genericity worry ("m can be dynamic and small"), not on a measurement. Production m is 65536,
  which is neither dynamic nor small -- so whether 512 or 1024 is faster HERE is simply unknown. That
  is exactly the shape of the gap that cost 2.2x on the gmm side, where `calculate_tiling` likewise
  picked a tile for reasons that did not apply.

  Same guard as before applies to any answer: the win must hold under a RAGGED router, because group
  sizes are runtime data.
  """
  label, size_m, num_groups, size_k, size_n = shape
  # tgmm's operands: lhs [m, k] (the activations) and rhs [m, n] (the incoming gradient);
  # the output is the weight gradient [g, k, n].
  k0, k1 = jax.random.split(jax.random.PRNGKey(2))
  lhs = jax.random.normal(k0, (size_m, size_k), jnp.bfloat16)
  rhs = jax.random.normal(k1, (size_m, size_n), jnp.bfloat16)
  gs = {d: group_sizes(d, size_m, num_groups) for d in GROUP_DISTS}
  if device is not None:
    lhs, rhs = jax.device_put(lhs, device), jax.device_put(rhs, device)
    gs = {d: jax.device_put(v, device) for d, v in gs.items()}
  flops = 2 * size_m * size_k * size_n

  def mk(tiles):
    kwargs = {} if tiles is None else {"tile_info": tiles}
    return jax.jit(
        lambda l, r, g: tgmm_lib.tgmm_v2(
            l, r, g, num_groups, preferred_element_type=jnp.bfloat16, **kwargs
        )
    )

  out = []
  # AUTO first (what calculate_tgmm_tiling actually returns today), then the forced ladder.
  configs = [None] + [
      gmm_lib.TileSizes(tile_m=tm, tile_k=size_k, tile_n=size_n // div)
      for tm in TGMM_TILE_MS
      for div in TGMM_TILE_N_DIVS
      if tm <= size_m and (size_n // div) >= 2 * pltpu.get_tpu_info().mxu_column_size
  ]
  for tiles in configs:
    tm_s = "AUTO" if tiles is None else str(tiles.tile_m)
    tn_s = "AUTO" if tiles is None else str(tiles.tile_n)
    fn = mk(tiles)
    for dist in GROUP_DISTS:
      key = (label, size_k, size_n, dist, "tgmm", tm_s, tn_s)
      try:
        us = _time_us(fn, lhs, rhs, gs[dist])
        out.append(key + (us, flops / (us * 1e-6) / 1e12, ""))
      except Exception as e:  # noqa: BLE001 - a non-admissible tile must not kill the sweep
        out.append(key + (float("nan"), float("nan"),
                          type(e).__name__ + ": " + str(e).replace("\n", " ")[:70]))
      if report is not None:
        report(out[-1])
  return out


def _init_backend() -> None:
  """Join the slice if we are one process of many; no-op on a single host.

  GKE hands each pod a slice-wide TPU topology, so on a multi-VM allocation the JAX TPU client has
  to be told the process layout before `jax.devices()` works -- that is what the trainer's
  setup_spmd does. Wrapped, because the same file must still run unchanged on a 1-process box
  (where initialize() has no coordinator to find) -- and because the py_binary that carries this
  into the trainer image may not depend on `requests`, which jax's GKE auto-detect imports. libtpu
  builds the slice from its own env either way, so a failure here is not fatal.
  """
  try:
    jax.distributed.initialize()
  except Exception as e:  # noqa: BLE001 - single-process is a legitimate outcome
    print(f"note: jax.distributed.initialize() skipped ({type(e).__name__}: {e})")


def main() -> int:
  _init_backend()
  if jax.default_backend() != "tpu":
    print("SKIP: needs a TPU (Pallas kernel; interpret mode cannot run this on jax 0.8.3).")
    return 0
  # One local device is all this measures; the rest of the slice idles (see MULTI-HOST NOTE).
  device = jax.local_devices()[0]
  proc = jax.process_index()
  # Every rank emits ONE line, so the log shows whether all of them got through init; only
  # rank 0 emits the table.
  print(f"[rank {proc}/{jax.process_count()}] device={device} "
        f"local={jax.local_device_count()} global={jax.device_count()}", flush=True)
  quiet = proc != 0

  info = pltpu.get_tpu_info()
  # gmm_v2's own default when vmem_limit_bytes is None.
  vmem_limit = int(info.vmem_capacity_bytes * 0.9)
  if not quiet:
    print(f"jax {jax.__version__} | {device.device_kind} | repeats={REPEATS}")
    print(f"tpu_info: vmem_capacity={info.vmem_capacity_bytes / 2**20:.1f} MiB "
          f"=> vmem_limit={vmem_limit / 2**20:.1f} MiB | "
          f"mxu_column_size={info.mxu_column_size} num_lanes={info.num_lanes}")
    for lab, m, g, k, n in SHAPES:
      print(f"  shape {lab} m={m} g={g} k={k} n={n} mean_rows_per_group={m // g}")
    print(f"{'shape':<16}{'k->n':<12}{'groups':<10}{'transp':<7}{'tile_m':>8}{'tile_n':>12}"
          f"{'us/call':>10}{'TFLOP/s':>8}  note")

  rows = []
  report = None if quiet else (lambda r: print(_fmt_row(r), flush=True))
  # DTYPE is the OUTER loop so the bf16 control rows land first: if they do not reproduce rounds
  # 1-6, the harness has drifted and nothing below it is trustworthy.
  for dt in DTYPES:
    if not quiet:
      print(f"--- dtypes: lhs={jnp.dtype(dt[0]).name} rhs={jnp.dtype(dt[1]).name} "
            f"(tag {dt[2]}) ---", flush=True)
    for shape in SHAPES:
      rows += run_shape(shape, vmem_limit, device, report, dt)

  # TGMM (drhs) -- production shapes only; this kernel's tiling has never been measured.
  if not quiet:
    print()
    print(f"{'shape':<16}{'k->n':<12}{'groups':<10}{'kern':<7}{'tile_m':>8}{'tile_n':>12}"
          f"{'us/call':>10}{'TFLOP/s':>8}  note")
  tgmm_rows = []
  for shape in SHAPES[:2]:
    tgmm_rows += run_tgmm_shape(shape, device, report)
  rows += tgmm_rows

  # Correctness only at the two production shapes: the question it answers (does tile_m change the
  # maths?) is shape-independent, and it is the expensive part of the run.
  correctness = []
  for size_k, size_n in DIRECTIONS:
    correctness.append(f"correctness {size_k}->{size_n}:")
    correctness += check_correctness(size_k, size_n, device)
  if quiet:
    return 0
  print()
  for line in correctness:
    print(line, flush=True)

  # Summary, per (shape, router): the transposed path's gain (already-shipped fix) AND the
  # untransposed path's headroom (the candidate lever -- raising tile_m for EVERY caller).
  print()
  ok = lambda r: r[7] == r[7]  # noqa: E731 - not NaN
  print(f"{'shape':<16}{'router':<10}| transposed: 128 -> best        "
        f"| untransposed: 128 -> best      | AUTO")
  for lab, _m, _g, size_k, _n in SHAPES:
    for dist in GROUP_DISTS:
      # `r[4] is True/False` for gmm rows and the string "tgmm" for tgmm rows -- filter
      # to bools, else `sorted()` below compares bool against str and raises TypeError
      # (it did: r4 measured everything, then died printing the summary).
      sel = [r for r in rows if r[0] == lab and r[3] == dist and ok(r)
             and isinstance(r[4], bool)]
      forced = [r for r in sel if not str(r[5]).startswith("AUTO")]
      auto = {r[4]: r for r in sel if str(r[5]).startswith("AUTO")}
      parts = []
      for transpose in (True, False):
        f = [r for r in forced if r[4] is transpose]
        base = min((r for r in f if r[5] == "128"), key=lambda r: r[7], default=None)
        best = min(f, key=lambda r: r[7], default=None)
        if base and best:
          parts.append(f"{base[7]:6.0f} -> {best[7]:6.0f} (tm={best[5]:>3}) x{base[7] / best[7]:.2f}")
        else:
          parts.append(" " * 30)
      a = " ".join(f"{'T' if t else 'F'}:tm={r[5].split(':')[1]}/{r[7]:.0f}us"
                   for t, r in sorted(auto.items(), reverse=True))
      print(f"{lab:<11}{dist:<10}| {parts[0]} | {parts[1]} | {a}")
  print()
  print("TGMM (drhs) -- is the hardcoded tile_m=256 the right choice at production m?")
  print(f"{'shape':<16}{'router':<10}| AUTO      | best forced           | verdict")
  for lab, _m, _g, size_k, _n in SHAPES[:2]:
    for dist in GROUP_DISTS:
      sel = [r for r in tgmm_rows if r[0] == lab and r[3] == dist and r[7] == r[7]]
      auto = next((r for r in sel if r[5] == "AUTO"), None)
      forced = [r for r in sel if r[5] != "AUTO"]
      if not auto or not forced:
        continue
      best = min(forced, key=lambda r: r[7])
      gain = auto[7] / best[7]
      verdict = (f"{gain:.2f}x faster at tile_m={best[5]}" if gain > 1.02
                 else "AUTO is already best (no lever)")
      print(f"{lab:<11}{dist:<10}| {auto[7]:6.0f} us | {best[7]:6.0f} us (tm={best[5]:>4}) | {verdict}")
  print("A tile_m that beats AUTO in EVERY router row is a real lever; one that only wins on the")
  print("balanced router is the tile_m=512 trap from the gmm side all over again.")

  print()
  print("Read the UNTRANSPOSED column: it is the same lever applied to the other 16 of 20 GMM calls "
        "per step. A tile_m that wins there in EVERY (shape, router) row is safe to make the default; "
        "one that loses anywhere is not, because calculate_tiling cannot see the router.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
