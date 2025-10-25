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

"""Networks for joint PPO + behavior cloning (L2T) training."""

from typing import Mapping, Sequence

import flax
from flax import linen
import jax

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks


@flax.struct.dataclass
class L2TNetworks:
  """Container holding teacher (PPO) and student policy networks."""

  teacher: ppo_networks.PPONetworks
  student_policy: networks.FeedForwardNetwork
  student_distribution: distribution.ParametricDistribution


def make_l2t_networks(
  observation_size: types.ObservationSize,
  action_size: int,
  preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
  teacher_policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
  teacher_value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
  student_policy_hidden_layer_sizes: Sequence[int] = (256, 256),
  activation: networks.ActivationFn = linen.swish,
  student_activation: networks.ActivationFn | None = None,
  teacher_policy_obs_key: str = "privileged_state",
  teacher_value_obs_key: str = "privileged_state",
  student_policy_obs_key: str = "state",
  distribution_type: str = "tanh_normal",
  noise_std_type: str = "scalar",
  init_noise_std: float = 1.0,
  state_dependent_std: bool = False,
  policy_network_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
  policy_network_kernel_init_kwargs: Mapping[str, float] | None = None,
  value_network_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
  value_network_kernel_init_kwargs: Mapping[str, float] | None = None,
  student_policy_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
  student_policy_kernel_init_kwargs: Mapping[str, float] | None = None,
) -> L2TNetworks:
  """Build PPO teacher and BC student networks sharing observation stats."""
  student_activation = student_activation or activation
  policy_kernel_kwargs = policy_network_kernel_init_kwargs or {}
  value_kernel_kwargs = value_network_kernel_init_kwargs or {}
  student_kernel_kwargs = student_policy_kernel_init_kwargs or {}

  teacher = ppo_networks.make_ppo_networks(
    observation_size,
    action_size,
    preprocess_observations_fn=preprocess_observations_fn,
    policy_hidden_layer_sizes=teacher_policy_hidden_layer_sizes,
    value_hidden_layer_sizes=teacher_value_hidden_layer_sizes,
    activation=activation,
    policy_obs_key=teacher_policy_obs_key,
    value_obs_key=teacher_value_obs_key,
    distribution_type=distribution_type,
    noise_std_type=noise_std_type,
    init_noise_std=init_noise_std,
    state_dependent_std=state_dependent_std,
    policy_network_kernel_init_fn=policy_network_kernel_init_fn,
    policy_network_kernel_init_kwargs=policy_kernel_kwargs,
    value_network_kernel_init_fn=value_network_kernel_init_fn,
    value_network_kernel_init_kwargs=value_kernel_kwargs,
  )

  if distribution_type == "normal":
    student_dist: distribution.ParametricDistribution = (
      distribution.NormalDistribution(event_size=action_size)
    )
  elif distribution_type == "tanh_normal":
    student_dist = distribution.NormalTanhDistribution(event_size=action_size)
  else:
    raise ValueError(
      "Unsupported distribution type for L2T student. "
      'Expected "normal" or "tanh_normal" but got '
      f"{distribution_type!r}."
    )

  student_policy = networks.make_policy_network(
    student_dist.param_size,
    observation_size,
    preprocess_observations_fn=preprocess_observations_fn,
    hidden_layer_sizes=student_policy_hidden_layer_sizes,
    activation=student_activation,
    obs_key=student_policy_obs_key,
    distribution_type=distribution_type,
    noise_std_type=noise_std_type,
    init_noise_std=init_noise_std,
    state_dependent_std=state_dependent_std,
    kernel_init=student_policy_kernel_init_fn(**student_kernel_kwargs),
  )

  return L2TNetworks(
    teacher=teacher,
    student_policy=student_policy,
    student_distribution=student_dist,
  )
