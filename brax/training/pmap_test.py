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

"""Tests for pmap helpers."""

from absl.testing import absltest
from brax.training import pmap
import jax
from jax import numpy as jnp


class PmapTest(absltest.TestCase):

  def test_bcast_local_devices_adds_local_device_axis(self):
    replicated = pmap.bcast_local_devices(
        {"x": jnp.array([1.0, 2.0]), "y": 3.0}, 1
    )

    self.assertEqual(replicated["x"].shape, (1, 2))
    self.assertEqual(replicated["y"].shape, (1,))
    self.assertEqual(float(replicated["y"][0]), 3.0)

  def test_bcast_local_devices_uses_requested_device_count(self):
    device_count = min(2, jax.local_device_count())
    replicated = pmap.bcast_local_devices(jnp.array([1.0, 2.0]), device_count)

    self.assertEqual(replicated.shape, (device_count, 2))
    self.assertTrue(bool(jnp.all(replicated == jnp.array([1.0, 2.0]))))


if __name__ == "__main__":
  absltest.main()
