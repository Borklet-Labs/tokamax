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
"""A library of accelerator kernels."""

# TEST BRANCH ONLY -- jax 0.8.3 compatibility shim. DO NOT send upstream.
#
# Upstream targets jax>=0.11, which is where jax.sharding.ManualAxisType was added. _src/ops/
# ragged_dot references it in 18 places, all but three as type annotations -- and none of those
# modules use `from __future__ import annotations`, so the annotations are evaluated eagerly and
# `import tokamax` dies at import time on jax 0.8.3 with
# "AttributeError: module 'jax.sharding' has no attribute 'ManualAxisType'".
#
# Importing any submodule runs this __init__ first, so the shim has to live here: there is no way
# to reach tokamax._src.ops.experimental.gmm_v2 without executing the eager chain below.
import jax.sharding as _jax_sharding

if not hasattr(_jax_sharding, "ManualAxisType"):

  class ManualAxisType:
    """Stand-in for the jax>=0.11 type; only ever used as an annotation on this branch."""

    def __init__(self, **kwargs):
      self.__dict__.update(kwargs)

  _jax_sharding.ManualAxisType = ManualAxisType

# pltpu.TensorCoreMesh is the only Pallas symbol the gmm_v2/tgmm_v2 kernels use that jax 0.8.3
# does not export (checked: 1 of the 21 they reference). It is a rename, not a missing feature --
# jax._src.pallas.mosaic.core.TensorCoreMesh exists in 0.8.3, but its constructor takes
# (devices, axis_names) while jax>=0.11 is called as TensorCoreMesh(axis_name=...). 0.8.3's
# create_tensorcore_mesh(axis_name) does exactly that derivation and returns the real class, so
# this forwards to it rather than stubbing anything out.
import jax.experimental.pallas.tpu as _pltpu

if not hasattr(_pltpu, "TensorCoreMesh"):

  def _tensorcore_mesh(axis_name, **kwargs):
    return _pltpu.create_tensorcore_mesh(axis_name, **kwargs)

  _pltpu.TensorCoreMesh = _tensorcore_mesh

# jax>=0.11 renamed two pl.kernel parameters: out_shape -> out_type and
# scratch_shapes -> scratch_types. The gmm_v2/tgmm_v2 kernels call the new spelling. Verified by
# inspect.signature against a real jax 0.8.3 install that this is the ONLY remaining signature
# mismatch -- BlockSpec, Buffered, CompilerParams, emit_pipeline, CostEstimate, SemaphoreType,
# make_async_copy and with_memory_space_constraint all accept what the call sites pass.
import inspect as _inspect

import jax.experimental.pallas as _pl

if "out_type" not in _inspect.signature(_pl.kernel).parameters:
  _orig_kernel = _pl.kernel

  def _kernel(*args, out_type=None, scratch_types=None, **kwargs):
    if out_type is not None:
      kwargs["out_shape"] = out_type
    if scratch_types is not None:
      kwargs["scratch_shapes"] = scratch_types
    return _orig_kernel(*args, **kwargs)

  _pl.kernel = _kernel

# pylint: disable=g-importing-member,useless-import-alias
from tokamax import autotuning as autotuning
from tokamax import benchmarking as benchmarking
from tokamax import config as config
from tokamax._src.autotuning.api import autotune as autotune
from tokamax._src.autotuning.api import AutotuningResult as AutotuningResult
from tokamax._src.batching import BatchedShapeDtype as BatchedShapeDtype
from tokamax._src.benchmarking import benchmark as benchmark
from tokamax._src.benchmarking import BenchmarkData as BenchmarkData
from tokamax._src.benchmarking import standardize_function as standardize_function
from tokamax._src.hlo_utils import DISABLE_JAX_EXPORT_CHECKS as DISABLE_JAX_EXPORT_CHECKS
from tokamax._src.ops.attention.api import dot_product_attention as dot_product_attention
from tokamax._src.ops.attention.api import Implementation as DotProductAttentionImplementation
from tokamax._src.ops.gated_linear_unit.api import gated_linear_unit as gated_linear_unit
from tokamax._src.ops.linear_softmax_cross_entropy_loss.api import linear_softmax_cross_entropy_loss as linear_softmax_cross_entropy_loss
from tokamax._src.ops.normalization.api import layer_norm as layer_norm
from tokamax._src.ops.op import BoundArguments as BoundArguments
from tokamax._src.ops.op import Op as Op
from tokamax._src.ops.ragged_dot.api import ragged_dot as ragged_dot
from tokamax._src.ops.ragged_dot.api import ragged_dot_general as ragged_dot_general
from tokamax._src.ops.ragged_dot.base import generate_group_sizes as generate_ragged_dot_group_sizes
from tokamax._src.ops.ragged_dot.base import GroupSizes as RaggedDotGroupSizes
from tokamax._src.ops.triangle_multiplication.api import triangle_multiplication as triangle_multiplication
from tokamax._src.version import TOKAMAX_VERSION as __version__
from tokamax._src.version import TOKAMAX_VERSION_INFO as __version_info__

# pylint: enable=g-importing-member,useless-import-alias
