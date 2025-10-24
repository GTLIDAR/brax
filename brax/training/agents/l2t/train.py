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
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

InferenceParams = Tuple[  # teacher params, student params
    Tuple[running_statistics.NestedMeanStd, Params, Params],
    Tuple[running_statistics.NestedMeanStd, Params],
]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


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
  return jax.tree_util.tree_map(lambda x: x[0], v)


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
      raise ValueError('Madrona-MJX requires a fixed batch size')
    if action_repeat != 1:
      raise ValueError(
          "Implement action_repeat using PipelineEnv's _n_frames to avoid"
          ' unnecessary rendering!'
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
    raise ValueError('episode_length must be specified in l2t.train')
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
          mode='edge',
      )
      return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

    out = {}
    for k_view, v_view in ub_obs.items():
      if k_view.startswith('pixels/'):
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
  return {k: v for k, v in obs.items() if not k.startswith('pixels/')}


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
        return actions, {'distribution_params': logits}
      raw_action = student_distribution.sample_no_postprocessing(
          logits, key_sample
      )
      log_prob = student_distribution.log_prob(logits, raw_action)
      actions = student_distribution.postprocess(raw_action)
      return actions, {
          'log_prob': log_prob,
          'raw_action': raw_action,
          'distribution_params': logits,
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
    learning_rate_schedule: Optional[
        Union[str, ppo_optimizer.LRSchedule]
    ] = None,
    network_factory: types.NetworkFactory[  # teacher + student networks
        l2t_networks.L2TNetworks
    ] = l2t_networks.make_l2t_networks,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing / restoring
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    run_evals: bool = True,
    # student specific
    student_learning_rate: Optional[float] = None,
    student_max_grad_norm: Optional[float] = None,
    student_bc_weight: float = 1.0,
):
  """Runs joint training of a PPO teacher and an L2 imitation student."""
  assert batch_size * num_minibatches % num_envs == 0
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
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
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
  key_teacher_policy, key_teacher_value, key_student_policy = jax.random.split(
      global_key, 3
  )

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
        reset_fn_donated_env_state, donate_argnums=(0,), keep_unused=True
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

  teacher_gradient_update_fn = gradients.gradient_update_fn(
      teacher_loss_fn,
      teacher_optimizer,
      pmap_axis_name=_PMAP_AXIS_NAME,
      has_aux=True,
  )

  def student_loss_fn(
      params: Params,
      normalizer_params: running_statistics.RunningStatisticsState,
      data: types.Transition,
      unused_key: PRNGKey,
  ):
    del unused_key
    logits = l2t_net.student_policy.apply(
        normalizer_params, params, data.observation
    )
    student_actions = l2t_net.student_distribution.mode(logits)
    diff = student_actions - data.action
    mse = jnp.mean(jnp.square(diff))
    loss = student_bc_weight * mse
    metrics = {
        'bc_loss': loss,
        'action_mse': mse,
    }
    return loss, metrics

  student_gradient_update_fn = gradients.gradient_update_fn(
      student_loss_fn,
      student_optimizer,
      pmap_axis_name=_PMAP_AXIS_NAME,
      has_aux=True,
  )

  metrics_aggregator = metric_logger.EpisodeMetricsLogger(
      steps_between_logging=training_metrics_steps
      or env_step_per_training_step,
      progress_fn=progress_fn,
  )

  def teacher_minibatch_step(
      carry,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), params, optimizer_state = teacher_gradient_update_fn(
        params,
        normalizer_params,
        data,
        key_loss,
        optimizer_state=optimizer_state,
    )
    metrics['learning_rate'] = jnp.array(learning_rate, dtype=float)
    if lr_is_adaptive_kl:
      kl_mean = metrics['kl_mean']
      kl_mean = jax.lax.pmean(kl_mean, axis_name=_PMAP_AXIS_NAME)
      optimizer_state, lr = ppo_optimizer.adaptive_kl_learning_rate(
          optimizer_state, kl_mean, desired_kl
      )
      metrics['learning_rate'] = lr
    return (optimizer_state, params, key), metrics

  def student_minibatch_step(
      carry,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params = carry
    (_, metrics), params, optimizer_state = student_gradient_update_fn(
        params,
        normalizer_params,
        data,
        jax.random.PRNGKey(0),
        optimizer_state=optimizer_state,
    )
    metrics['learning_rate'] = jnp.array(student_lr, dtype=float)
    return (optimizer_state, params), metrics

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
            student_minibatch_step, normalizer_params=student_norm
        ),
        (student_optimizer_state, student_params),
        shuffled_data,
        length=num_minibatches,
    )

    student_metrics = {f'student/{k}': v for k, v in student_metrics.items()}
    metrics = {**teacher_metrics, **student_metrics}

    return (
        teacher_optimizer_state,
        teacher_params,
        key,
        student_optimizer_state,
        student_params,
    ), metrics

  def training_step(
      carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
  ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
    training_state, state, key = carry
    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

    teacher_policy = teacher_make_policy((
        training_state.teacher.normalizer_params,
        training_state.teacher.params.policy,
        training_state.teacher.params.value,
    ))

    def f(carry, unused_t):
      current_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      next_state, data = acting.generate_unroll(
          env,
          current_state,
          teacher_policy,
          current_key,
          unroll_length,
          extra_fields=('truncation', 'episode_metrics', 'episode_done'),
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
          data.extras['state_extras']['episode_metrics'],
          data.extras['state_extras']['episode_done'],
          metrics,
      )

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
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    return training_state, env_state, metrics

  teacher_init_params = ppo_losses.PPONetworkParams(
      policy=l2t_net.teacher.policy_network.init(key_teacher_policy),
      value=l2t_net.teacher.value_network.init(key_teacher_value),
  )
  student_init_params = l2t_net.student_policy.init(key_student_policy)

  obs_specs = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')),
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
    return (
        policy_wrapper,
        _unpmap(_pack_params(init_training_state)),
        {},
    )

  training_state = jax.device_put_replicated(
      init_training_state, jax.local_devices()[:local_devices_to_use]
  )

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
  evaluator = acting.Evaluator(
      eval_env,
      functools.partial(
          policy_wrapper,
          deterministic=deterministic_eval,
      ),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
  )

  training_metrics = {}
  training_walltime = 0
  current_step = 0

  def host_make_policy(params, deterministic=False, agent='student'):
    return policy_wrapper(params, deterministic=deterministic, agent=agent)

  params = _unpmap(_pack_params(training_state))
  policy_params_fn(current_step, host_make_policy, params)

  metrics = {}
  if process_id == 0 and num_evals > 1 and run_evals:
    metrics = evaluator.run_evaluation(
        params,
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  num_evals_after_init = max(num_evals - 1, 1)

  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

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
        env_state = reset_fn((training_state, env_state), key_envs)

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
        metrics = evaluator.run_evaluation(
            params,
            training_metrics,
        )
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  if total_steps < num_timesteps:
    raise AssertionError(
        f'Total steps {total_steps} is less than'
        f' `num_timesteps`={num_timesteps}.'
    )
  pmap.assert_is_replicated(training_state)
  params = _unpmap(_pack_params(training_state))
  logging.info('total steps: %s', total_steps)
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
      agent: str = 'student',
  ) -> types.Policy:
    teacher_params, student_params = params
    if agent == 'teacher':
      return teacher_make_policy_fn(teacher_params, deterministic=deterministic)
    if agent == 'student':
      return student_make_policy_fn(student_params, deterministic=deterministic)
    raise ValueError(
        f'Unsupported agent: {agent}. Choose "teacher" or "student".'
    )

  return make_policy
