# Copyright 2025 The Brax Authors and Feiyang Wu (feiyangwu@gatech.edu).
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

"""Checkpoint utilities for L2T teacher/student training."""

from typing import Any, Tuple, Union

from etils import epath
import jax
from ml_collections import config_dict
import numpy as np
from orbax import checkpoint as ocp

from brax.training import checkpoint
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.l2t import networks as l2t_networks


_CONFIG_FNAME = "l2t_network_config.json"

InferenceParams = Tuple[
  Tuple[running_statistics.NestedMeanStd, Any, Any],
  Tuple[running_statistics.NestedMeanStd, Any],
]


def save(
  path: Union[str, epath.Path],
  step: int,
  params: InferenceParams,
  config: config_dict.ConfigDict,
):
  """Persists joint teacher/student parameters and config."""
  return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def network_config(
  observation_size: types.ObservationSize,
  action_size: int,
  normalize_observations: bool,
  network_factory: types.NetworkFactory[l2t_networks.L2TNetworks],
) -> config_dict.ConfigDict:
  """Creates a config for reconstructing networks from a checkpoint."""
  return checkpoint.network_config(
    observation_size,
    action_size,
    normalize_observations,
    network_factory,
  )


def load_config(path: Union[str, epath.Path]) -> config_dict.ConfigDict:
  """Loads the network config stored alongside checkpoint."""
  path = epath.Path(path)
  config_path = path / _CONFIG_FNAME
  return checkpoint.load_config(config_path)


def load(path: Union[str, epath.Path]) -> InferenceParams:
  """Loads L2T parameters from disk."""
  path = epath.Path(path)
  if not path.exists():
    raise ValueError(f"checkpoint path does not exist: {path.as_posix()}")

  metadata = ocp.PyTreeCheckpointer().metadata(path).item_metadata
  restore_args = jax.tree.map(
    lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata
  )
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  target = orbax_checkpointer.restore(
    path, ocp.args.PyTreeRestore(restore_args=restore_args), item=None
  )

  teacher_norm, teacher_policy, teacher_value = target[0]
  student_norm, student_policy = target[1]

  teacher_norm = running_statistics.RunningStatisticsState(**teacher_norm)
  student_norm = running_statistics.RunningStatisticsState(**student_norm)

  return (
    (teacher_norm, teacher_policy, teacher_value),
    (student_norm, student_policy),
  )
