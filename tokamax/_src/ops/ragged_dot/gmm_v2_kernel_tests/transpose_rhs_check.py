# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Numerical validation of `gmm_v2`'s `transpose_rhs`.

Proves the identity the optimization rests on -- consuming the weight with its
last two axes reinterpreted by the BlockSpec index map computes the same thing
as physically swapping the weight and using the normal path:

    gmm_v2(lhs, w_swapped, transpose_rhs=True)
    == gmm_v2(lhs, w,      transpose_rhs=False)
    == reference ragged matmul

That is exactly how a `dlhs` backward call reuses the forward weight without
paying a `.swapaxes(1, 2)` HBM copy.

Run it on a TPU -- that is the real check, and it needs no special setup:

    python -m tokamax._src.ops.ragged_dot.gmm_v2_kernel_tests.transpose_rhs_check

On a CPU-only host it falls back to Pallas's TPU interpret mode. NOTE that the
fallback does NOT work on jax 0.8.3: that version's interpret emulator cannot
handle this kernel's reshaped-ref DMAs (`AttributeError: 'RefReshaper' object
has no attribute 'indices'` from `pallas/mosaic/interpret/utils.py:to_range`).
That is a pre-existing limitation of the emulator, not of this kernel -- the
unmodified upstream kernel fails there identically -- so on jax 0.8.3 use a
TPU. Exits non-zero on any mismatch.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_gmm_kernel as gmm_lib


def reference_ragged_matmul(lhs, rhs, group_sizes):
  """Plain-jnp ragged matmul: rows of `lhs` grouped by `group_sizes`."""
  out = np.zeros((lhs.shape[0], rhs.shape[2]), dtype=np.float32)
  start = 0
  for g, n_rows in enumerate(np.asarray(group_sizes)):
    end = start + int(n_rows)
    if end > start:
      out[start:end] = np.asarray(
          jnp.dot(
              lhs[start:end].astype(jnp.float32), rhs[g].astype(jnp.float32)
          )
      )
    start = end
  return out


def _enable_cpu_fallback():
  """Points the TPU device probes at TPU7x so the kernel can run on CPU.

  Only touches this process; the kernel source is untouched. Returns the
  interpret-mode context manager to run under.
  """
  from jax._src.pallas.mosaic import core as mosaic_core  # pylint: disable=g-import-not-at-top
  from jax.experimental.pallas import tpu as pltpu  # pylint: disable=g-import-not-at-top

  mosaic_core.get_device_kind = lambda: "TPU7x"
  mosaic_core.get_num_device_cores = lambda: 1
  # The dual-core branch launches over `create_tensorcore_mesh`, which reads
  # `devices[0].num_cores` -- absent on CPU devices. Supply it.
  orig = pltpu.create_tensorcore_mesh
  pltpu.create_tensorcore_mesh = lambda axis_name, **kw: orig(
      axis_name, num_cores=kw.get("num_cores") or 1
  )
  return pltpu.force_tpu_interpret_mode()


def _report(name, got, want, tol):
  err = float(np.max(np.abs(got.astype(np.float32) - want.astype(np.float32))))
  ok = err <= tol
  print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} max_abs_err={err:.3e}")
  return ok


def main() -> int:
  on_tpu = jax.devices()[0].platform == "tpu"
  print(f"jax {jax.__version__} | devices={jax.devices()} | "
        f"{'REAL TPU' if on_tpu else 'CPU + interpret fallback'}")

  import contextlib  # pylint: disable=g-import-not-at-top
  ctx = contextlib.nullcontext() if on_tpu else _enable_cpu_fallback()

  # size_k != size_n on purpose: that inequality is what makes a double
  # transpose detectable, and it is the real case (k_model != n_ffn).
  num_groups, size_k, size_n, size_m = 2, 256, 128, 64
  k1, k2 = jax.random.split(jax.random.PRNGKey(0))
  lhs = jax.random.normal(k1, (size_m, size_k), jnp.bfloat16)
  w = jax.random.normal(k2, (num_groups, size_k, size_n), jnp.bfloat16)
  group_sizes = jnp.array([32, 32], jnp.int32)  # sums to size_m

  want = reference_ragged_matmul(lhs, w, group_sizes)
  tol = 0.05 * max(1.0, float(np.max(np.abs(want))))

  oks = []
  with ctx:
    # 1. The untouched path (transpose_rhs defaults to False).
    out_normal = gmm_lib.gmm_v2(
        lhs=lhs, rhs=w, group_sizes=group_sizes,
        preferred_element_type=jnp.float32,
    )
    oks.append(_report("flag=False, rhs=[G,k,n] (untouched path)",
                       np.asarray(out_normal), want, tol))

    # 2. THE IDENTITY: same contraction, weight's last two axes swapped and
    #    reinterpreted by the index map rather than physically transposed.
    out_transposed = gmm_lib.gmm_v2(
        lhs=lhs, rhs=w.swapaxes(1, 2), group_sizes=group_sizes,
        preferred_element_type=jnp.float32, transpose_rhs=True,
    )
    oks.append(_report("flag=True,  rhs=[G,n,k] (index-map transpose)",
                       np.asarray(out_transposed), want, tol))

    # 3. The two paths must agree with each other, not just the reference.
    oks.append(_report("flag=True == flag=False",
                       np.asarray(out_transposed), np.asarray(out_normal), tol))

    # 4. Uneven groups: exercises the per-group index map together with
    #    partial-tile K masking on the transposed layout.
    uneven = jnp.array([40, 24], jnp.int32)
    out_uneven = gmm_lib.gmm_v2(
        lhs=lhs, rhs=w.swapaxes(1, 2), group_sizes=uneven,
        preferred_element_type=jnp.float32, transpose_rhs=True,
    )
    oks.append(_report("flag=True, uneven groups [40,24]",
                       np.asarray(out_uneven),
                       reference_ragged_matmul(lhs, w, uneven), tol))

  # 5. Guards must reject unimplemented combinations, not compute a wrong
  #    answer. These raise during tracing, so no device work is needed.
  for label, kwargs in (
      ("rhs_bias",
       dict(rhs_bias=jnp.zeros((num_groups, 1, size_k), jnp.bfloat16))),
      ("fuse_act", dict(fuse_act="silu")),
  ):
    try:
      gmm_lib.gmm_v2(
          lhs=lhs, rhs=w.swapaxes(1, 2), group_sizes=group_sizes,
          transpose_rhs=True, **kwargs,
      )
      print(f"  FAIL  guard did NOT fire for transpose_rhs + {label}")
      oks.append(False)
    except NotImplementedError:
      print(f"  PASS  guard rejects transpose_rhs + {label}")
      oks.append(True)

  print(f"\n{'ALL CHECKS PASSED' if all(oks) else 'FAILURES PRESENT'}"
        f"  ({sum(oks)}/{len(oks)})")
  return 0 if all(oks) else 1


if __name__ == "__main__":
  sys.exit(main())
