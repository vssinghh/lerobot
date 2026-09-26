# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""`eval_policy_all` reports success counts and Wilson intervals at every aggregation level."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts import lerobot_eval  # noqa: E402

_SUCCESSES = {
    ("suite_a", 0): [True] * 9 + [False],
    ("suite_a", 1): [True] * 5 + [False] * 5,
    ("suite_b", 0): [False] * 10,
}


class _DummyEnv:
    def close(self):
        pass


def _fake_run_one(task_group, task_id, env, **kwargs):
    successes = _SUCCESSES[(task_group, task_id)]
    metrics = {
        "sum_rewards": [1.0] * len(successes),
        "max_rewards": [1.0] * len(successes),
        "successes": list(successes),
        "video_paths": [],
        "predicted_video_paths": [],
    }
    return task_group, task_id, metrics


@pytest.fixture
def info(monkeypatch):
    monkeypatch.setattr(lerobot_eval, "run_one", _fake_run_one)
    envs = {"suite_a": {0: _DummyEnv(), 1: _DummyEnv()}, "suite_b": {0: _DummyEnv()}}
    policy = MagicMock()
    policy.training = False
    return lerobot_eval.eval_policy_all(envs, policy, None, None, None, None, n_episodes=10)


def test_per_task_entries_carry_counts_and_interval(info):
    task = next(t for t in info["per_task"] if (t["task_group"], t["task_id"]) == ("suite_a", 0))
    assert task["n_episodes"] == 10
    assert task["n_success"] == 9
    assert task["pc_success"] == pytest.approx(90.0)
    assert task["pc_success_ci95"] == pytest.approx([59.58, 98.21], abs=0.05)


def test_per_group_aggregate_pools_its_tasks(info):
    group = info["per_group"]["suite_a"]
    assert group["n_episodes"] == 20
    assert group["n_success"] == 14
    assert group["pc_success"] == pytest.approx(70.0)
    assert group["pc_success_ci95"] == pytest.approx([48.10, 85.45], abs=0.05)


def test_overall_aggregate_pools_every_task(info):
    overall = info["overall"]
    assert overall["n_episodes"] == 30
    assert overall["n_success"] == 14
    assert overall["pc_success"] == pytest.approx(100 * 14 / 30)
    assert overall["pc_success_ci95"] == pytest.approx([30.23, 63.86], abs=0.05)


def test_existing_keys_are_untouched(info):
    overall = info["overall"]
    for key in ("avg_sum_reward", "avg_max_reward", "eval_s", "eval_ep_s", "video_paths"):
        assert key in overall
    assert "pc_success_control" not in overall
    assert "fisher_p_value" not in overall
    assert set(info) == {"per_task", "per_group", "overall"}


def test_run_control_attaches_matched_seed_control_stats_and_fisher_p(monkeypatch):
    ctrl_successes = {
        ("suite_a", 0): [True] * 10,
        ("suite_a", 1): [True] * 9 + [False],
        ("suite_b", 0): [True] * 5 + [False] * 5,
    }
    primary_prep = object()
    control_prep = object()
    recorded_calls = []

    class _TrackedEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def _dual_run_one(task_group, task_id, env, **kwargs):
        assert not env.closed, "control pass must run before env.close()"
        is_control = kwargs.get("preprocessor") is control_prep
        recorded_calls.append((task_group, task_id, is_control, kwargs.get("start_seed")))
        table = ctrl_successes if is_control else _SUCCESSES
        succ = table[(task_group, task_id)]
        metrics = {
            "sum_rewards": [1.0] * len(succ),
            "max_rewards": [1.0] * len(succ),
            "successes": list(succ),
            "video_paths": [],
            "predicted_video_paths": [],
        }
        return task_group, task_id, metrics

    monkeypatch.setattr(lerobot_eval, "run_one", _dual_run_one)
    envs = {
        "suite_a": {0: _TrackedEnv(), 1: _TrackedEnv()},
        "suite_b": {0: _TrackedEnv()},
    }
    policy = MagicMock()
    policy.training = False

    out = lerobot_eval.eval_policy_all(
        envs,
        policy,
        None,
        None,
        primary_prep,
        None,
        n_episodes=10,
        start_seed=42,
        run_control=True,
        control_preprocessor=control_prep,
    )

    # Every task ran both primary and control with identical start_seed=42
    assert len(recorded_calls) == 6
    assert all(seed == 42 for _, _, _, seed in recorded_calls)

    # Per-task control stats: suite_b task 0 has 0/10 primary vs 5/10 control
    task_b0 = next(t for t in out["per_task"] if (t["task_group"], t["task_id"]) == ("suite_b", 0))
    assert task_b0["n_episodes_control"] == 10
    assert task_b0["n_success_control"] == 5
    assert task_b0["pc_success_control"] == pytest.approx(50.0)
    assert task_b0["pc_success_control_ci95"] == pytest.approx([23.66, 76.34], abs=0.05)
    assert 0.0 < task_b0["fisher_p_value"] < 0.05

    # Per-group control stats: suite_a has 14/20 primary vs 19/20 control
    group_a = out["per_group"]["suite_a"]
    assert group_a["n_episodes_control"] == 20
    assert group_a["n_success_control"] == 19
    assert group_a["pc_success_control"] == pytest.approx(95.0)
    assert 0.0 < group_a["fisher_p_value"] < 0.15

    # Overall control stats: 14/30 primary vs 24/30 control
    overall = out["overall"]
    assert overall["n_episodes_control"] == 30
    assert overall["n_success_control"] == 24
    assert overall["pc_success_control"] == pytest.approx(80.0)
    assert 0.0 < overall["fisher_p_value"] < 0.05
