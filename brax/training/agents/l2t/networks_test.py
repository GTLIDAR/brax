# Copyright 2026 The Brax Authors.
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

"""Tests for L2T networks."""

from absl.testing import absltest
from brax.training.acme import running_statistics
from brax.training.agents.l2t import networks as l2t_networks
import jax
from jax import numpy as jnp


class L2TNetworksTest(absltest.TestCase):

  def test_make_l2t_networks_with_teacher_and_student_obs_keys(self):
    obs_size = {"state": (8,), "privileged_state": (12,)}
    net = l2t_networks.make_l2t_networks(
        obs_size,
        3,
        teacher_policy_hidden_layer_sizes=(4,),
        teacher_value_hidden_layer_sizes=(4,),
        student_policy_hidden_layer_sizes=(4,),
    )
    obs = {
        "state": jnp.ones((8,)),
        "privileged_state": jnp.ones((12,)),
    }
    normalizer = running_statistics.init_state(obs)
    teacher_policy_params = net.teacher.policy_network.init(jax.random.PRNGKey(0))
    teacher_value_params = net.teacher.value_network.init(jax.random.PRNGKey(1))
    student_policy_params = net.student_policy.init(jax.random.PRNGKey(2))

    teacher_logits = net.teacher.policy_network.apply(
        normalizer, teacher_policy_params, obs
    )
    teacher_value = net.teacher.value_network.apply(
        normalizer, teacher_value_params, obs
    )
    student_logits = net.student_policy.apply(normalizer, student_policy_params, obs)

    self.assertEqual(teacher_logits.shape, (6,))
    self.assertEqual(teacher_value.shape, ())
    self.assertEqual(student_logits.shape, (6,))


if __name__ == "__main__":
  absltest.main()
