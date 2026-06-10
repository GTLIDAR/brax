# Copyright 2025 The Brax Authors.
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

"""Joint PPO teacher and L2 behavior-cloning student training."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.l2t import checkpoint as l2t_checkpoint
from brax.training.agents.l2t import networks as l2t_networks
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import optimizer as ppo_optimizer
from brax.training.types import Params
from brax.training.types import PRNGKey


InferenceParams = Tuple[  # teacher params, student params
  Tuple[running_statistics.NestedMeanStd, Params, Params],
  Tuple[running_statistics.NestedMeanStd, Params],
]
Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"
_MIXED_ACTION_LOG_PROB_FLOOR = -jnp.inf


@flax.struct.dataclass
class TeacherTrainingState:
  optimizer_state: optax.OptState
  params: ppo_losses.PPONetworkParams
  normalizer_params: running_statistics.RunningStatisticsState


@flax.struct.dataclass
class StudentTrainingState:
  optimizer_state: optax.OptState
  params: Params
  normalizer_params: running_statistics.RunningStatisticsState


@flax.struct.dataclass
class TrainingState:
  teacher: TeacherTrainingState
  student: StudentTrainingState
  env_steps: types.UInt64


def _unpmap(v):
  def _unpack(x):
    # Handle 0-dimensional arrays (scalars)
    if x.ndim == 0:
      return x
    # Handle replicated arrays (first dimension is device dimension)
    if x.ndim > 0:
      return x[0]
    return x

  return jax.tree_util.tree_map(_unpack, v)


def _strip_weak_type(tree):
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return jnp.astype(leaf, leaf.dtype)

  return jax.tree_util.tree_map(f, tree)


def _validate_madrona_args(
  madrona_backend: bool,
  num_envs: int,
  num_eval_envs: int,
  action_repeat: int,
  eval_env: Optional[envs.Env] = None,
):
  if madrona_backend:
    if eval_env:
      raise ValueError("Madrona-MJX doesn't support multiple env instances")
    if num_eval_envs != num_envs:
      raise ValueError("Madrona-MJX requires a fixed batch size")
    if action_repeat != 1:
      raise ValueError(
        "Implement action_repeat using PipelineEnv's _n_frames to avoid"
        " unnecessary rendering!"
      )


def _maybe_wrap_env(
  env: envs.Env,
  wrap_env: bool,
  num_envs: int,
  episode_length: Optional[int],
  action_repeat: int,
  device_count: int,
  key_env: PRNGKey,
  wrap_env_fn: Optional[Callable[[Any], Any]] = None,
  randomization_fn: Optional[
    Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
  ] = None,
):
  if not wrap_env:
    return env
  if episode_length is None:
    raise ValueError("episode_length must be specified in l2t.train")
  v_randomization_fn = None
  if randomization_fn is not None:
    randomization_batch_size = num_envs // device_count
    randomization_rng = jax.random.split(key_env, randomization_batch_size)
    v_randomization_fn = functools.partial(
      randomization_fn, rng=randomization_rng
    )
  wrap_for_training = wrap_env_fn or envs.training.wrap
  env = wrap_for_training(
    env,
    episode_length=episode_length,
    action_repeat=action_repeat,
    randomization_fn=v_randomization_fn,
  )  # pytype: disable=wrong-keyword-args
  return env


def _random_translate_pixels(
  obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
  @jax.vmap
  def rt_all_views(
    ub_obs: Mapping[str, jax.Array], key: PRNGKey
  ) -> Mapping[str, jax.Array]:
    def rt_view(img: jax.Array, padding: int, key: PRNGKey) -> jax.Array:
      crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
      zero = jnp.zeros((1,), dtype=jnp.int32)
      crop_from = jnp.concatenate([zero, crop_from, zero])
      padded_img = jnp.pad(
        img,
        ((0, 0), (padding, padding), (padding, padding), (0, 0)),
        mode="edge",
      )
      return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

    out = {}
    for k_view, v_view in ub_obs.items():
      if k_view.startswith("pixels/"):
        key, key_shift = jax.random.split(key)
        out[k_view] = rt_view(v_view, 4, key_shift)
    return {**ub_obs, **out}

  bdim = next(iter(obs.items()), None)[1].shape[0]
  keys = jax.random.split(key, bdim)
  obs = rt_all_views(obs, keys)
  return obs


def _remove_pixels(
  obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
  if not isinstance(obs, Mapping):
    return obs
  return {k: v for k, v in obs.items() if not k.startswith("pixels/")}


def _batch_size_from_observation(observation: types.Observation) -> int:
  leaves = jax.tree_util.tree_leaves(observation)
  if not leaves:
    raise ValueError("Cannot infer batch size from an empty observation tree.")
  return leaves[0].shape[0]


def _where_with_batch_mask(
  mask: jax.Array, teacher_value: jax.Array, student_value: jax.Array
) -> jax.Array:
  while mask.ndim < teacher_value.ndim:
    mask = mask[..., None]
  return jnp.where(mask, teacher_value, student_value)


def _uint64_to_float(value: types.UInt64) -> jax.Array:
  hi = jnp.asarray(value.hi, dtype=jnp.float32)
  lo = jnp.asarray(value.lo, dtype=jnp.float32)
  return hi * (2.0**32) + lo


def _merge_l2t_eval_metrics(
  teacher_metrics: Metrics,
  student_metrics: Metrics,
  training_metrics: Metrics,
  teacher_num_eval_envs: int,
  student_num_eval_envs: int,
  episode_length: Optional[int],
  num_eval_envs: int,
) -> Metrics:
  metrics = dict(training_metrics)
  teacher_eval_metrics = {
    k: v for k, v in teacher_metrics.items() if k.startswith("eval/")
  }
  student_eval_metrics = {
    k: v for k, v in student_metrics.items() if k.startswith("eval/")
  }
  for key, value in teacher_eval_metrics.items():
    metrics[f"eval/teacher/{key[len('eval/'):]}"] = value
  for key, value in student_eval_metrics.items():
    metrics[f"eval/student/{key[len('eval/'):]}"] = value

  for key, teacher_value in teacher_eval_metrics.items():
    if key not in student_eval_metrics:
      continue
    student_value = student_eval_metrics[key]
    if key in ("eval/walltime", "eval/epoch_eval_time"):
      metrics[key] = teacher_value + student_value
    elif key != "eval/sps":
      metrics[key] = 0.5 * (teacher_value + student_value)
  epoch_eval_time = metrics.get("eval/epoch_eval_time")
  if epoch_eval_time is not None and episode_length is not None:
    metrics["eval/sps"] = episode_length * num_eval_envs / epoch_eval_time
  metrics["eval/teacher/num_envs"] = teacher_num_eval_envs
  metrics["eval/student/num_envs"] = student_num_eval_envs
  return metrics


def _make_student_inference_fn(
  student_network: types.NetworkFactory[Any],
  student_distribution: Any,
):
  def make_policy(
    params: Tuple[running_statistics.NestedMeanStd, Params],
    deterministic: bool = False,
  ) -> types.Policy:
    normalizer_params, policy_params = params

    def policy(
      observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      logits = student_network.apply(
        normalizer_params, policy_params, observations
      )
      if deterministic:
        actions = student_distribution.mode(logits)
        return actions, {"distribution_params": logits}
      raw_action = student_distribution.sample_no_postprocessing(
        logits, key_sample
      )
      log_prob = student_distribution.log_prob(logits, raw_action)
      actions = student_distribution.postprocess(raw_action)
      return actions, {
        "log_prob": log_prob,
        "raw_action": raw_action,
        "distribution_params": logits,
      }

    return policy

  return make_policy


def _pack_params(training_state: TrainingState) -> InferenceParams:
  return (
    (
      training_state.teacher.normalizer_params,
      training_state.teacher.params.policy,
      training_state.teacher.params.value,
    ),
    (
      training_state.student.normalizer_params,
      training_state.student.params,
    ),
  )


def train(
  environment: envs.Env,
  num_timesteps: int,
  max_devices_per_host: Optional[int] = None,
  wrap_env: bool = True,
  madrona_backend: bool = False,
  augment_pixels: bool = False,
  # environment wrapper
  num_envs: int = 1,
  episode_length: Optional[int] = None,
  action_repeat: int = 1,
  wrap_env_fn: Optional[Callable[[Any], Any]] = None,
  randomization_fn: Optional[
    Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
  ] = None,
  # teacher PPO params
  learning_rate: float = 1e-4,
  entropy_cost: float = 1e-4,
  discounting: float = 0.9,
  unroll_length: int = 10,
  batch_size: int = 32,
  num_minibatches: int = 16,
  num_updates_per_batch: int = 2,
  num_resets_per_eval: int = 0,
  normalize_observations: bool = False,
  reward_scaling: float = 1.0,
  clipping_epsilon: float = 0.3,
  gae_lambda: float = 0.95,
  max_grad_norm: Optional[float] = None,
  normalize_advantage: bool = True,
  vf_loss_coefficient: float = 0.5,
  desired_kl: float = 0.01,
  learning_rate_schedule: Optional[Union[str, ppo_optimizer.LRSchedule]] = None,
  network_factory: types.NetworkFactory[  # teacher + student networks
    l2t_networks.L2TNetworks
  ] = l2t_networks.make_l2t_networks,
  seed: int = 0,
  use_pmap_on_reset: bool = True,
  # eval
  num_evals: int = 1,
  eval_env: Optional[envs.Env] = None,
  num_eval_envs: int = 256,
  deterministic_eval: bool = False,
  fixed_eval_rng: bool = False,
  # training metrics
  log_training_metrics: bool = False,
  training_metrics_steps: Optional[int] = None,
  # callbacks
  progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
  policy_params_fn: Callable[..., None] = lambda *args: None,
  # checkpointing / restoring
  save_checkpoint_path: Optional[str] = None,
  restore_checkpoint_path: Optional[str] = None,
  restore_teacher_params: Optional[Any] = None,
  restore_params: Optional[Any] = None,
  restore_value_fn: bool = True,
  run_evals: bool = True,
  # student specific
  student_learning_rate: Optional[float] = None,
  student_max_grad_norm: Optional[float] = None,
  student_bc_weight: float = 1.0,
  student_use_nll_loss: bool = True,
  student_entropy_cost: float = 0.0,
  student_use_huber_loss: bool = False,
  student_huber_delta: float = 1.0,
  student_action_mse_weight: float = 0.0,
  student_reference_action_mse_weight: float = 0.0,
  student_reference_action_obs_key: str = "state",
  student_reference_action_slice: Optional[Tuple[int, int]] = None,
  student_match_distribution_params: bool = False,
  student_ppo_weight: float = 0.0,
  student_clone_teacher_mode: bool = True,
  teacher_sampling_start_probability: float = 1.0,
  teacher_sampling_end_probability: float = 0.8,
  teacher_sampling_warmup_steps: int = 0,
):
  """Runs joint training of a PPO teacher and an L2 imitation student."""
  assert batch_size * num_minibatches % num_envs == 0
  if run_evals and num_evals > 0:
    if num_eval_envs < 2:
      raise ValueError("L2T eval requires at least 2 envs to split agents.")
    if num_eval_envs % 2:
      raise ValueError("L2T eval requires an even num_eval_envs.")
  if not 0.0 <= teacher_sampling_start_probability <= 1.0:
    raise ValueError("teacher_sampling_start_probability must be in [0, 1].")
  if not 0.0 <= teacher_sampling_end_probability <= 1.0:
    raise ValueError("teacher_sampling_end_probability must be in [0, 1].")
  if teacher_sampling_warmup_steps < 0:
    raise ValueError("teacher_sampling_warmup_steps must be non-negative.")
  if student_action_mse_weight < 0.0:
    raise ValueError("student_action_mse_weight must be non-negative.")
  if student_reference_action_mse_weight < 0.0:
    raise ValueError(
      "student_reference_action_mse_weight must be non-negative."
    )
  if (
    student_reference_action_mse_weight > 0.0
    and student_reference_action_slice is None
  ):
    raise ValueError(
      "student_reference_action_slice is required when "
      "student_reference_action_mse_weight is positive."
    )
  if student_ppo_weight < 0.0:
    raise ValueError("student_ppo_weight must be non-negative.")
  _validate_madrona_args(
    madrona_backend, num_envs, num_eval_envs, action_repeat, eval_env
  )

  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
    "Device count: %d, process count: %d (id %d), local device count: %d, "
    "devices to be used count: %d",
    jax.device_count(),
    process_count,
    process_id,
    local_device_count,
    local_devices_to_use,
  )
  device_count = local_devices_to_use * process_count

  env_step_per_training_step = (
    batch_size * unroll_length * num_minibatches * action_repeat
  )
  num_evals_after_init = max(num_evals - 1, 1)
  num_training_steps_per_epoch = np.ceil(
    num_timesteps
    / (
      num_evals_after_init
      * env_step_per_training_step
      * max(num_resets_per_eval, 1)
    )
  ).astype(int)

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  # Keep the teacher initialization bit-for-bit aligned with PPO for the same
  # seed.  The student gets an independent key derived after the PPO split.
  key_teacher_policy, key_teacher_value = jax.random.split(global_key)
  key_student_policy = jax.random.fold_in(global_key, 1)

  assert num_envs % device_count == 0

  env = _maybe_wrap_env(
    environment,
    wrap_env,
    num_envs,
    episode_length,
    action_repeat,
    device_count,
    key_env,
    wrap_env_fn,
    randomization_fn,
  )

  def reset_fn_donated_env_state(env_state_donated, key_envs):
    return env.reset(key_envs)

  def vmap_reset_fn_donated_env_state(env_state_donated, key_envs):
    return jax.vmap(env.reset)(key_envs)

  key_envs = jax.random.split(key_env, num_envs // process_count)
  key_envs = jnp.reshape(
    key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
  )
  if local_devices_to_use > 1 or use_pmap_on_reset:
    reset_fn_ = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    env_state = reset_fn_(key_envs)
    reset_fn = jax.pmap(
      reset_fn_donated_env_state,
      axis_name=_PMAP_AXIS_NAME,
      donate_argnums=(0,),
    )
  else:
    reset_fn_ = jax.jit(jax.vmap(env.reset))
    env_state = reset_fn_(key_envs)
    reset_fn = jax.jit(
      vmap_reset_fn_donated_env_state, donate_argnums=(0,), keep_unused=True
    )

  obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  l2t_net = network_factory(
    obs_shape, env.action_size, preprocess_observations_fn=normalize
  )
  teacher_make_policy = ppo_networks.make_inference_fn(l2t_net.teacher)
  student_make_policy = _make_student_inference_fn(
    l2t_net.student_policy, l2t_net.student_distribution
  )
  policy_wrapper = _make_policy_wrapper(
    teacher_make_policy, student_make_policy
  )

  teacher_base_optimizer = optax.adam(learning_rate=learning_rate)
  lr_schedule = learning_rate_schedule or ppo_optimizer.LRSchedule.NONE
  lr_schedule = ppo_optimizer.LRSchedule(lr_schedule)
  lr_is_adaptive_kl = lr_schedule == ppo_optimizer.LRSchedule.ADAPTIVE_KL
  if lr_is_adaptive_kl:
    teacher_base_optimizer = optax.inject_hyperparams(optax.adam)(
      learning_rate=learning_rate
    )
  if max_grad_norm is not None:
    teacher_optimizer = optax.chain(
      optax.clip_by_global_norm(max_grad_norm),
      teacher_base_optimizer,
    )
  else:
    teacher_optimizer = teacher_base_optimizer

  student_lr = student_learning_rate or learning_rate
  student_optimizer: optax.GradientTransformation = optax.adam(
    learning_rate=student_lr
  )
  if student_max_grad_norm is not None:
    student_optimizer = optax.chain(
      optax.clip_by_global_norm(student_max_grad_norm),
      student_optimizer,
    )

  teacher_loss_fn = functools.partial(
    ppo_losses.compute_ppo_loss,
    ppo_network=l2t_net.teacher,
    entropy_cost=entropy_cost,
    discounting=discounting,
    reward_scaling=reward_scaling,
    gae_lambda=gae_lambda,
    clipping_epsilon=clipping_epsilon,
    normalize_advantage=normalize_advantage,
    vf_coefficient=vf_loss_coefficient,
  )

  teacher_loss_and_pgrad_fn = gradients.loss_and_pgrad(
    teacher_loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
  )

  def student_loss_fn(
    params: Params,
    normalizer_params: running_statistics.RunningStatisticsState,
    data: types.Transition,
    key_loss: PRNGKey,
    teacher_normalizer_params: running_statistics.RunningStatisticsState,
    teacher_value_params: Params,
  ):
    logits = l2t_net.student_policy.apply(
      normalizer_params, params, data.observation
    )

    policy_extras = data.extras["policy_extras"]
    teacher_action = policy_extras.get("teacher_action", data.action)
    if student_clone_teacher_mode:
      teacher_action = policy_extras.get("teacher_mode_action", teacher_action)

    # Action-based losses
    student_actions = l2t_net.student_distribution.mode(logits)
    diff = student_actions - teacher_action

    if student_use_huber_loss:
      # Huber loss: more robust to outliers than MSE
      squared_diff = jnp.square(diff)
      abs_diff = jnp.abs(diff)
      action_loss = jnp.where(
        abs_diff < student_huber_delta,
        0.5 * squared_diff,
        student_huber_delta * (abs_diff - 0.5 * student_huber_delta),
      )
      action_loss = jnp.mean(action_loss)
    else:
      # Standard MSE loss
      action_loss = jnp.mean(jnp.square(diff))

    reference_action_loss = jnp.array(0.0)
    reference_action_mse = jnp.array(0.0)
    if student_reference_action_mse_weight > 0.0:
      start, end = student_reference_action_slice
      if isinstance(data.observation, Mapping):
        reference_obs = data.observation[student_reference_action_obs_key]
      else:
        reference_obs = data.observation
      reference_action = jax.lax.stop_gradient(reference_obs[..., start:end])
      reference_action_mse = jnp.mean(
        jnp.square(student_actions - reference_action)
      )
      reference_action_loss = (
        student_reference_action_mse_weight * reference_action_mse
      )

    # Negative log-likelihood loss (more principled for probabilistic policies)
    # Note: Requires teacher to generate raw_action in policy_extras
    nll_loss = 0.0
    if student_use_nll_loss:
      teacher_raw_action = policy_extras.get(
        "teacher_raw_action", policy_extras["raw_action"]
      )
      if student_clone_teacher_mode:
        teacher_raw_action = policy_extras.get(
          "teacher_mode_raw_action", teacher_raw_action
        )
      # Use raw action for NLL computation (before postprocessing)
      nll = -l2t_net.student_distribution.log_prob(logits, teacher_raw_action)
      nll_loss = jnp.mean(nll)

    # Distribution parameter matching (for better learning of uncertainty)
    # Note: Requires teacher to generate distribution_params in policy_extras
    dist_param_loss = 0.0
    if student_match_distribution_params:
      student_dist = l2t_net.student_distribution.create_dist(logits)
      if hasattr(student_dist, "loc") and hasattr(student_dist, "scale"):
        # Get teacher distribution params
        teacher_dist_params = policy_extras["distribution_params"]
        teacher_dist = l2t_net.student_distribution.create_dist(
          teacher_dist_params
        )
        if hasattr(teacher_dist, "loc") and hasattr(teacher_dist, "scale"):
          # Match mean and std separately
          mean_diff = student_dist.loc - teacher_dist.loc
          std_diff = student_dist.scale - teacher_dist.scale
          dist_param_loss = jnp.mean(jnp.square(mean_diff)) + jnp.mean(
            jnp.square(std_diff)
          )

    # Entropy regularization (encourage exploration)
    entropy_loss = 0.0
    entropy = 0.0
    if student_entropy_cost > 0.0:
      entropy = jnp.mean(
        l2t_net.student_distribution.entropy(logits, key_loss)
      )
      entropy_loss = -student_entropy_cost * entropy

    # Combine losses
    if student_use_nll_loss:
      bc_loss = nll_loss
    else:
      bc_loss = action_loss
    action_mse_loss = student_action_mse_weight * action_loss

    ppo_actor_loss = jnp.array(0.0)
    ppo_entropy = jnp.array(0.0)
    ppo_entropy_loss = jnp.array(0.0)
    ppo_mask_mean = jnp.array(0.0)
    if student_ppo_weight > 0.0:
      baseline = l2t_net.teacher.value_network.apply(
        teacher_normalizer_params, teacher_value_params, data.observation
      )
      terminal_obs = jax.tree_util.tree_map(
        lambda x: x[-1], data.next_observation
      )
      bootstrap_value = l2t_net.teacher.value_network.apply(
        teacher_normalizer_params, teacher_value_params, terminal_obs
      )
      baseline = jax.lax.stop_gradient(baseline)
      bootstrap_value = jax.lax.stop_gradient(bootstrap_value)
      rewards = data.reward * reward_scaling
      truncation = data.extras["state_extras"]["truncation"]
      termination = (1 - data.discount) * (1 - truncation)
      _, advantages = ppo_losses.compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
      )
      if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (
          advantages.std() + 1e-8
        )
      selected_branch_log_prob = policy_extras.get(
        "branch_log_prob", policy_extras["log_prob"]
      )
      target_action_log_probs = l2t_net.student_distribution.log_prob(
        logits, policy_extras["raw_action"]
      )
      log_rho_s = target_action_log_probs - selected_branch_log_prob
      log_rho_s = jnp.nan_to_num(
        log_rho_s, nan=0.0, neginf=-20.0, posinf=20.0
      )
      log_rho_s = jnp.clip(log_rho_s, -20.0, 20.0)
      rho_s = jnp.exp(log_rho_s)
      surrogate_loss1 = rho_s * advantages
      surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon)
        * advantages
      )
      surrogate_loss = jnp.minimum(surrogate_loss1, surrogate_loss2)
      student_policy_mask = 1.0 - policy_extras.get(
        "sampled_teacher", jnp.ones_like(surrogate_loss)
      )
      student_policy_mask = student_policy_mask.astype(surrogate_loss.dtype)
      masked_surrogate = jnp.where(
        student_policy_mask > 0.0, surrogate_loss, 0.0
      )
      mask_sum = jnp.sum(student_policy_mask)
      ppo_actor_loss = -jnp.sum(masked_surrogate) / jnp.maximum(mask_sum, 1.0)
      student_entropy = l2t_net.student_distribution.entropy(logits, key_loss)
      masked_entropy = jnp.where(
        student_policy_mask > 0.0, student_entropy, 0.0
      )
      ppo_entropy = jnp.sum(masked_entropy) / jnp.maximum(mask_sum, 1.0)
      ppo_entropy_loss = -student_entropy_cost * ppo_entropy
      ppo_mask_mean = jnp.mean(student_policy_mask)

    total_loss = (
      student_bc_weight * bc_loss
      + action_mse_loss
      + reference_action_loss
      + dist_param_loss
      + entropy_loss
      + student_ppo_weight * (ppo_actor_loss + ppo_entropy_loss)
    )

    metrics = {
      "total_loss": total_loss,
      "bc_loss": bc_loss,
      "action_mse": action_loss,
      "action_mse_loss": action_mse_loss,
      "action_mse_weight": jnp.array(student_action_mse_weight),
      "reference_action_mse": reference_action_mse,
      "reference_action_mse_loss": reference_action_loss,
      "reference_action_mse_weight": jnp.array(
        student_reference_action_mse_weight
      ),
      "nll_loss": nll_loss if student_use_nll_loss else jnp.array(0.0),
      "dist_param_loss": dist_param_loss,
      "entropy": entropy,
      "entropy_loss": entropy_loss,
      "ppo_actor_loss": ppo_actor_loss,
      "ppo_entropy": ppo_entropy,
      "ppo_entropy_loss": ppo_entropy_loss,
      "ppo_mask_mean": ppo_mask_mean,
    }
    return total_loss, metrics

  student_gradient_update_fn = gradients.gradient_update_fn(
    student_loss_fn,
    student_optimizer,
    pmap_axis_name=_PMAP_AXIS_NAME,
    has_aux=True,
  )

  metrics_aggregator = metric_logger.EpisodeMetricsLogger(
    steps_between_logging=training_metrics_steps or env_step_per_training_step,
    progress_fn=progress_fn,
  )

  def teacher_minibatch_step(
    carry,
    data: types.Transition,
    normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), grads = teacher_loss_and_pgrad_fn(
      params, normalizer_params, data, key_loss
    )
    metrics["learning_rate"] = jnp.array(learning_rate, dtype=float)
    if lr_is_adaptive_kl:
      kl_mean = metrics["kl_mean"]
      kl_mean = jax.lax.pmean(kl_mean, axis_name=_PMAP_AXIS_NAME)
      optimizer_state, lr = ppo_optimizer.adaptive_kl_learning_rate(
        optimizer_state, kl_mean, desired_kl
      )
      metrics["learning_rate"] = lr
    params_update, optimizer_state = teacher_optimizer.update(
      grads, optimizer_state
    )
    params = optax.apply_updates(params, params_update)
    return (optimizer_state, params, key), metrics

  def student_minibatch_step(
    carry,
    data: types.Transition,
    normalizer_params: running_statistics.RunningStatisticsState,
    teacher_normalizer_params: running_statistics.RunningStatisticsState,
    teacher_value_params: Params,
  ):
    optimizer_state, params = carry
    (_, metrics), params, optimizer_state = student_gradient_update_fn(
      params,
      normalizer_params,
      data,
      jax.random.PRNGKey(0),
      teacher_normalizer_params,
      teacher_value_params,
      optimizer_state=optimizer_state,
    )
    metrics["learning_rate"] = jnp.array(student_lr, dtype=float)
    return (optimizer_state, params), metrics

  def teacher_sample_probability(env_steps: types.UInt64) -> jax.Array:
    schedule_steps = max(
      float(num_timesteps - teacher_sampling_warmup_steps), 1.0
    )
    scheduled_env_steps = jnp.maximum(
      _uint64_to_float(env_steps) - float(teacher_sampling_warmup_steps),
      0.0,
    )
    progress = scheduled_env_steps / schedule_steps
    progress = jnp.clip(progress, 0.0, 1.0)
    return (
      teacher_sampling_start_probability
      + (teacher_sampling_end_probability - teacher_sampling_start_probability)
      * progress
    )

  def sgd_step(
    carry,
    unused_t,
    data: types.Transition,
    teacher_norm: running_statistics.RunningStatisticsState,
    student_norm: running_statistics.RunningStatisticsState,
  ):
    (
      teacher_optimizer_state,
      teacher_params,
      key,
      student_optimizer_state,
      student_params,
    ) = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    sgd_data = data
    if augment_pixels:
      key, key_rt = jax.random.split(key)
      r_translate = functools.partial(_random_translate_pixels, key=key_rt)
      sgd_data = types.Transition(
        observation=r_translate(data.observation),
        action=data.action,
        reward=data.reward,
        discount=data.discount,
        next_observation=r_translate(data.next_observation),
        extras=data.extras,
      )

    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    shuffled_data = jax.tree_util.tree_map(convert_data, sgd_data)

    (teacher_optimizer_state, teacher_params, _), teacher_metrics = (
      jax.lax.scan(
        functools.partial(
          teacher_minibatch_step, normalizer_params=teacher_norm
        ),
        (teacher_optimizer_state, teacher_params, key_grad),
        shuffled_data,
        length=num_minibatches,
      )
    )

    (student_optimizer_state, student_params), student_metrics = jax.lax.scan(
      functools.partial(
        student_minibatch_step,
        normalizer_params=student_norm,
        teacher_normalizer_params=teacher_norm,
        teacher_value_params=teacher_params.value,
      ),
      (student_optimizer_state, student_params),
      shuffled_data,
      length=num_minibatches,
    )

    student_metrics = {f"student/{k}": v for k, v in student_metrics.items()}
    metrics = {**teacher_metrics, **student_metrics}

    return (
      teacher_optimizer_state,
      teacher_params,
      key,
      student_optimizer_state,
      student_params,
    ), metrics

  def make_mixed_rollout_policy(
    training_state: TrainingState, probability_teacher: jax.Array
  ) -> types.Policy:
    teacher_policy = teacher_make_policy(
      (
        training_state.teacher.normalizer_params,
        training_state.teacher.params.policy,
        training_state.teacher.params.value,
      )
    )
    student_policy = student_make_policy(
      (
        training_state.student.normalizer_params,
        training_state.student.params,
      )
    )

    def policy(observations: types.Observation, key_sample: PRNGKey):
      key_teacher, key_student, key_mix = jax.random.split(key_sample, 3)
      teacher_action, teacher_extras = teacher_policy(
        observations, key_teacher
      )
      student_action, student_extras = student_policy(
        observations, key_student
      )
      batch_size = _batch_size_from_observation(observations)
      sampled_teacher = jax.random.bernoulli(
        key_mix, probability_teacher, (batch_size,)
      )
      action = _where_with_batch_mask(
        sampled_teacher, teacher_action, student_action
      )

      teacher_distribution_params = teacher_extras["distribution_params"]
      student_distribution_params = student_extras["distribution_params"]
      teacher_raw_action = teacher_extras["raw_action"]
      student_raw_action = student_extras["raw_action"]
      teacher_log_prob_for_teacher_raw = teacher_extras["log_prob"]
      student_log_prob_for_student_raw = student_extras["log_prob"]
      teacher_mode_raw_action = (
        l2t_net.teacher.parametric_action_distribution.create_dist(
          teacher_distribution_params
        ).mode()
      )
      teacher_mode_action = (
        l2t_net.teacher.parametric_action_distribution.postprocess(
          teacher_mode_raw_action
        )
      )
      teacher_log_prob_for_student_raw = (
        l2t_net.teacher.parametric_action_distribution.log_prob(
          teacher_distribution_params, student_raw_action
        )
      )
      student_log_prob_for_teacher_raw = (
        l2t_net.student_distribution.log_prob(
          student_distribution_params, teacher_raw_action
        )
      )

      def clean_log_prob(log_prob: jax.Array) -> jax.Array:
        return jnp.where(
          jnp.isfinite(log_prob), log_prob, _MIXED_ACTION_LOG_PROB_FLOOR
        )

      teacher_log_prob_for_teacher_raw = clean_log_prob(
        teacher_log_prob_for_teacher_raw
      )
      student_log_prob_for_student_raw = clean_log_prob(
        student_log_prob_for_student_raw
      )
      teacher_log_prob_for_student_raw = clean_log_prob(
        teacher_log_prob_for_student_raw
      )
      student_log_prob_for_teacher_raw = clean_log_prob(
        student_log_prob_for_teacher_raw
      )
      branch_log_prob = jnp.where(
        sampled_teacher,
        teacher_log_prob_for_teacher_raw,
        student_log_prob_for_student_raw,
      )
      selected_teacher_log_prob = jnp.where(
        sampled_teacher,
        teacher_log_prob_for_teacher_raw,
        teacher_log_prob_for_student_raw,
      )
      selected_student_log_prob = jnp.where(
        sampled_teacher,
        student_log_prob_for_teacher_raw,
        student_log_prob_for_student_raw,
      )
      log_teacher_weight = jnp.log(probability_teacher)
      log_student_weight = jnp.log1p(-probability_teacher)
      behavior_log_prob = jnp.logaddexp(
        log_teacher_weight + selected_teacher_log_prob,
        log_student_weight + selected_student_log_prob,
      )
      behavior_log_prob = clean_log_prob(behavior_log_prob)
      policy_extras = {
        "log_prob": behavior_log_prob,
        "branch_log_prob": branch_log_prob,
        "raw_action": _where_with_batch_mask(
          sampled_teacher, teacher_raw_action, student_raw_action
        ),
        "teacher_action": teacher_action,
        "teacher_raw_action": teacher_raw_action,
        "teacher_mode_action": teacher_mode_action,
        "teacher_mode_raw_action": teacher_mode_raw_action,
        "distribution_params": teacher_distribution_params,
        "sampled_teacher": sampled_teacher.astype(jnp.float32),
        "policy_gradient_mask": sampled_teacher.astype(jnp.float32),
        "teacher_sample_probability": jnp.full(
          (batch_size,), probability_teacher, dtype=jnp.float32
        ),
      }
      return action, policy_extras

    return policy

  def make_teacher_rollout_policy(training_state: TrainingState) -> types.Policy:
    return teacher_make_policy(
      (
        training_state.teacher.normalizer_params,
        training_state.teacher.params.policy,
        training_state.teacher.params.value,
      )
    )

  use_teacher_only_rollout = (
    teacher_sampling_start_probability == 1.0
    and teacher_sampling_end_probability == 1.0
  )

  def training_step(
    carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
  ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
    training_state, state, key = carry
    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)
    probability_teacher = teacher_sample_probability(training_state.env_steps)
    if use_teacher_only_rollout:
      rollout_policy = make_teacher_rollout_policy(training_state)
    else:
      rollout_policy = make_mixed_rollout_policy(
        training_state, probability_teacher
      )

    def f(carry, unused_t):
      current_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      next_state, data = acting.generate_unroll(
        env,
        current_state,
        rollout_policy,
        current_key,
        unroll_length,
        extra_fields=("truncation", "episode_metrics", "episode_done"),
      )
      return (next_state, next_key), data

    (state, _), data = jax.lax.scan(
      f,
      (state, key_generate_unroll),
      (),
      length=batch_size * num_minibatches // num_envs,
    )
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
    )

    teacher_normalizer_params = training_state.teacher.normalizer_params
    if not lr_is_adaptive_kl:
      teacher_normalizer_params = running_statistics.update(
        teacher_normalizer_params,
        _remove_pixels(data.observation),
        pmap_axis_name=_PMAP_AXIS_NAME,
      )
    student_normalizer_params = running_statistics.update(
      training_state.student.normalizer_params,
      _remove_pixels(data.observation),
      pmap_axis_name=_PMAP_AXIS_NAME,
    )

    (opt_state, metrics) = jax.lax.scan(
      functools.partial(
        sgd_step,
        data=data,
        teacher_norm=teacher_normalizer_params,
        student_norm=student_normalizer_params,
      ),
      (
        training_state.teacher.optimizer_state,
        training_state.teacher.params,
        key_sgd,
        training_state.student.optimizer_state,
        training_state.student.params,
      ),
      (),
      length=num_updates_per_batch,
    )

    (
      teacher_optimizer_state,
      teacher_params,
      _,
      student_optimizer_state,
      student_params,
    ) = opt_state

    if lr_is_adaptive_kl:
      teacher_normalizer_params = running_statistics.update(
        teacher_normalizer_params,
        _remove_pixels(data.observation),
        pmap_axis_name=_PMAP_AXIS_NAME,
      )

    new_training_state = TrainingState(
      teacher=TeacherTrainingState(
        optimizer_state=teacher_optimizer_state,
        params=teacher_params,
        normalizer_params=teacher_normalizer_params,
      ),
      student=StudentTrainingState(
        optimizer_state=student_optimizer_state,
        params=student_params,
        normalizer_params=student_normalizer_params,
      ),
      env_steps=training_state.env_steps + env_step_per_training_step,
    )

    if log_training_metrics:
      jax.debug.callback(
        metrics_aggregator.update_episode_metrics,
        data.extras["state_extras"]["episode_metrics"],
        data.extras["state_extras"]["episode_done"],
        metrics,
      )

    metrics = {
      **metrics,
      "rollout/teacher_sample_probability": probability_teacher,
      "rollout/teacher_sample_fraction": jnp.array(1.0)
      if use_teacher_only_rollout
      else jnp.mean(data.extras["policy_extras"]["sampled_teacher"]),
    }
    return (new_training_state, state, new_key), metrics

  def training_epoch(
    training_state: TrainingState, state: envs.State, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, Metrics]:
    (training_state, state, _), loss_metrics = jax.lax.scan(
      training_step,
      (training_state, state, key),
      (),
      length=num_training_steps_per_epoch,
    )
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    return training_state, state, loss_metrics

  training_epoch = jax.pmap(
    training_epoch,
    axis_name=_PMAP_AXIS_NAME,
    donate_argnums=(0, 1),
  )

  def training_epoch_with_timing(
    training_state: TrainingState, env_state: envs.State, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, Metrics]:
    nonlocal training_walltime
    t = time.time()
    training_state, env_state = _strip_weak_type((training_state, env_state))
    result = training_epoch(training_state, env_state, key)
    training_state, env_state, metrics = _strip_weak_type(result)

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
      num_training_steps_per_epoch
      * env_step_per_training_step
      * max(num_resets_per_eval, 1)
    ) / epoch_training_time
    metrics = {
      "training/sps": sps,
      "training/walltime": training_walltime,
      **{f"training/{name}": value for name, value in metrics.items()},
    }
    return training_state, env_state, metrics

  teacher_init_params = ppo_losses.PPONetworkParams(
    policy=l2t_net.teacher.policy_network.init(key_teacher_policy),
    value=l2t_net.teacher.value_network.init(key_teacher_value),
  )
  student_init_params = l2t_net.student_policy.init(key_student_policy)

  obs_specs = jax.tree_util.tree_map(
    lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")),
    env_state.obs,
  )
  init_training_state = TrainingState(
    teacher=TeacherTrainingState(
      optimizer_state=teacher_optimizer.init(teacher_init_params),
      params=teacher_init_params,
      normalizer_params=running_statistics.init_state(
        _remove_pixels(obs_specs)
      ),
    ),
    student=StudentTrainingState(
      optimizer_state=student_optimizer.init(student_init_params),
      params=student_init_params,
      normalizer_params=running_statistics.init_state(
        _remove_pixels(obs_specs)
      ),
    ),
    env_steps=types.UInt64(hi=0, lo=0),
  )

  if restore_checkpoint_path is not None:
    restored = l2t_checkpoint.load(restore_checkpoint_path)
    teacher_value = (
      restored[0][2] if restore_value_fn else teacher_init_params.value
    )
    init_training_state = init_training_state.replace(
      teacher=init_training_state.teacher.replace(
        normalizer_params=restored[0][0],
        params=init_training_state.teacher.params.replace(
          policy=restored[0][1],
          value=teacher_value,
        ),
      ),
      student=init_training_state.student.replace(
        normalizer_params=restored[1][0],
        params=restored[1][1],
      ),
    )

  if restore_teacher_params is not None:
    teacher_value = (
      restore_teacher_params[2]
      if restore_value_fn
      else teacher_init_params.value
    )
    init_training_state = init_training_state.replace(
      teacher=init_training_state.teacher.replace(
        normalizer_params=restore_teacher_params[0],
        params=init_training_state.teacher.params.replace(
          policy=restore_teacher_params[1],
          value=teacher_value,
        ),
      ),
    )

  if restore_params is not None:
    teacher_value = (
      restore_params[0][2] if restore_value_fn else teacher_init_params.value
    )
    init_training_state = init_training_state.replace(
      teacher=init_training_state.teacher.replace(
        normalizer_params=restore_params[0][0],
        params=init_training_state.teacher.params.replace(
          policy=restore_params[0][1],
          value=teacher_value,
        ),
      ),
      student=init_training_state.student.replace(
        normalizer_params=restore_params[1][0],
        params=restore_params[1][1],
      ),
    )

  if num_timesteps == 0:
    # When num_timesteps == 0, state is not replicated, so don't use _unpmap.
    params = _pack_params(init_training_state)
    metrics = {}
    if process_id == 0 and run_evals and num_evals > 0:
      teacher_num_eval_envs = num_eval_envs // 2
      student_num_eval_envs = num_eval_envs - teacher_num_eval_envs
      eval_env = _maybe_wrap_env(
        eval_env or environment,
        wrap_env,
        num_eval_envs,
        episode_length,
        action_repeat,
        device_count=1,
        key_env=eval_key,
        wrap_env_fn=wrap_env_fn,
        randomization_fn=randomization_fn,
      )
      teacher_eval_key, student_eval_key = jax.random.split(eval_key)
      teacher_evaluator = acting.Evaluator(
        eval_env,
        functools.partial(
          policy_wrapper,
          deterministic=deterministic_eval,
          agent="teacher",
        ),
        num_eval_envs=teacher_num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=teacher_eval_key,
        fixed_key=fixed_eval_rng,
      )
      student_evaluator = acting.Evaluator(
        eval_env,
        functools.partial(
          policy_wrapper,
          deterministic=deterministic_eval,
          agent="student",
        ),
        num_eval_envs=student_num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=student_eval_key,
        fixed_key=fixed_eval_rng,
      )
      teacher_metrics = teacher_evaluator.run_evaluation(params, {})
      student_metrics = student_evaluator.run_evaluation(params, {})
      metrics = _merge_l2t_eval_metrics(
        teacher_metrics,
        student_metrics,
        {},
        teacher_num_eval_envs,
        student_num_eval_envs,
        episode_length,
        num_eval_envs,
      )
      logging.info(metrics)
      progress_fn(0, metrics)
    return (
      policy_wrapper,
      params,
      metrics,
    )

  training_state = pmap.bcast_local_devices(
    init_training_state, local_devices_to_use
  )

  teacher_evaluator = None
  student_evaluator = None
  teacher_num_eval_envs = num_eval_envs // 2
  student_num_eval_envs = num_eval_envs - teacher_num_eval_envs
  if run_evals:
    eval_env = _maybe_wrap_env(
      eval_env or environment,
      wrap_env,
      num_eval_envs,
      episode_length,
      action_repeat,
      device_count=1,
      key_env=eval_key,
      wrap_env_fn=wrap_env_fn,
      randomization_fn=randomization_fn,
    )
    teacher_eval_key, student_eval_key = jax.random.split(eval_key)
    teacher_evaluator = acting.Evaluator(
      eval_env,
      functools.partial(
        policy_wrapper,
        deterministic=deterministic_eval,
        agent="teacher",
      ),
      num_eval_envs=teacher_num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=teacher_eval_key,
      fixed_key=fixed_eval_rng,
    )
    student_evaluator = acting.Evaluator(
      eval_env,
      functools.partial(
        policy_wrapper,
        deterministic=deterministic_eval,
        agent="student",
      ),
      num_eval_envs=student_num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=student_eval_key,
      fixed_key=fixed_eval_rng,
    )

  training_metrics = {}
  training_walltime = 0
  current_step = 0

  def host_make_policy(params, deterministic=False, agent="student"):
    return policy_wrapper(params, deterministic=deterministic, agent=agent)

  params = _unpmap(_pack_params(training_state))
  policy_params_fn(current_step, host_make_policy, params)

  def run_l2t_evaluation(
    params: InferenceParams, training_metrics: Metrics
  ) -> Metrics:
    if teacher_evaluator is None or student_evaluator is None:
      return training_metrics
    teacher_metrics = teacher_evaluator.run_evaluation(params, {})
    student_metrics = student_evaluator.run_evaluation(params, {})
    return _merge_l2t_eval_metrics(
      teacher_metrics,
      student_metrics,
      training_metrics,
      teacher_num_eval_envs,
      student_num_eval_envs,
      episode_length,
      num_eval_envs,
    )

  metrics = {}
  if process_id == 0 and num_evals > 1 and run_evals:
    metrics = run_l2t_evaluation(params, {})
    logging.info(metrics)
    progress_fn(0, metrics)

  num_evals_after_init = max(num_evals - 1, 1)

  for it in range(num_evals_after_init):
    logging.info("starting iteration %s %s", it, time.time() - xt)

    for _ in range(max(num_resets_per_eval, 1)):
      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
      (training_state, env_state, training_metrics) = (
        training_epoch_with_timing(training_state, env_state, epoch_keys)
      )
      current_step = int(_unpmap(training_state.env_steps))

      key_envs = jax.vmap(
        lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
      )(key_envs, key_envs.shape[1])
      if num_resets_per_eval > 0:
        env_state = reset_fn(env_state, key_envs)

    if process_id != 0:
      continue

    params = _unpmap(_pack_params(training_state))

    policy_params_fn(current_step, host_make_policy, params)

    if save_checkpoint_path is not None:
      ckpt_config = l2t_checkpoint.network_config(
        observation_size=obs_specs,
        action_size=env.action_size,
        normalize_observations=normalize_observations,
        network_factory=network_factory,
      )
      l2t_checkpoint.save(
        save_checkpoint_path,
        current_step,
        params,
        ckpt_config,
      )

    if num_evals > 0:
      metrics = training_metrics
      if run_evals:
        metrics = run_l2t_evaluation(params, training_metrics)
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  if total_steps < num_timesteps:
    raise AssertionError(
      f"Total steps {total_steps} is less than `num_timesteps`={num_timesteps}."
    )
  pmap.assert_is_replicated(training_state)
  params = _unpmap(_pack_params(training_state))
  logging.info("total steps: %s", total_steps)
  pmap.synchronize_hosts()
  return (
    host_make_policy,
    params,
    metrics,
  )


def _make_policy_wrapper(
  teacher_make_policy_fn: Callable[..., types.Policy],
  student_make_policy_fn: Callable[..., types.Policy],
) -> Callable[..., types.Policy]:
  def make_policy(
    params: InferenceParams,
    deterministic: bool = False,
    agent: str = "teacher",
  ) -> types.Policy:
    teacher_params, student_params = params
    if agent == "teacher":
      return teacher_make_policy_fn(teacher_params, deterministic=deterministic)
    if agent == "student":
      return student_make_policy_fn(student_params, deterministic=deterministic)
    raise ValueError(
      f'Unsupported agent: {agent}. Choose "teacher" or "student".'
    )

  return make_policy
