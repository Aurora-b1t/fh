import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter


ROOT_DIR = Path(__file__).resolve().parents[1]
SPECIAL_DIR = ROOT_DIR / "special_hopping_test"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SPECIAL_DIR))

import offline_replay
import SAC as main_sac_module
import SAC_test as special_sac_module
from SAC import SAC as MainSAC
from SAC_test import ReplayBuffer as SpecialReplayBuffer
from SAC_test import SAC as SpecialSAC
from train_test import add_block_transitions as add_special_block_transitions


class ExtraEncodingTests(unittest.TestCase):
    MODULES = (main_sac_module, special_sac_module)

    def test_block_index_is_one_hot_encoded(self):
        raw_extra = torch.tensor(
            [[10.0, 0.0], [505.0, 4.0], [1000.0, 9.0]],
            dtype=torch.float32,
        )

        for module in self.MODULES:
            with self.subTest(module=module.__name__):
                encoded = module.normalize_extra(raw_extra)
                self.assertEqual(tuple(encoded.shape), (3, 11))
                torch.testing.assert_close(
                    encoded[:, 0], torch.tensor([-1.0, 0.0, 1.0])
                )
                torch.testing.assert_close(
                    encoded[:, 1:],
                    torch.nn.functional.one_hot(
                        torch.tensor([0, 4, 9]), num_classes=10
                    ).to(torch.float32),
                )

    def test_block_index_must_be_in_range(self):
        for module in self.MODULES:
            for block_idx in (-1.0, 10.0):
                raw_extra = torch.tensor(
                    [[100.0, block_idx]], dtype=torch.float32
                )
                with self.subTest(module=module.__name__, value=block_idx):
                    with self.assertRaises(RuntimeError):
                        module.normalize_extra(raw_extra)


class BlockTransitionTests(unittest.TestCase):
    def setUp(self):
        self.state = np.full((2, 3), 1.0, dtype=np.float32)
        self.next_state = np.full((2, 3), 2.0, dtype=np.float32)
        self.offsets = np.arange(10, dtype=np.int64)
        self.rewards = np.linspace(-1.0, 1.0, 10, dtype=np.float32)

    def assert_sequential_transitions(
        self,
        transitions,
        current_hoprate,
        next_hoprate,
    ):
        for block_idx, transition in enumerate(transitions):
            self.assertEqual(int(transition[2]), block_idx)
            self.assertEqual(transition[3], int(self.offsets[block_idx]))
            self.assertAlmostEqual(transition[4], float(self.rewards[block_idx]))
            self.assertEqual(int(transition[7]), (block_idx + 1) % 10)
            self.assertFalse(transition[8])

            if block_idx < 9:
                np.testing.assert_array_equal(transition[5], self.state)
                self.assertEqual(transition[6], current_hoprate)
            else:
                np.testing.assert_array_equal(transition[5], self.next_state)
                self.assertEqual(transition[6], next_hoprate)

    def test_shared_helper_builds_sequential_transitions(self):
        buffer = offline_replay.OfflineReplayBuffer(capacity=10)
        added = offline_replay.add_block_transitions(
            buffer,
            self.state,
            self.next_state,
            100.0,
            self.offsets,
            self.rewards,
            next_hoprate=200.0,
        )

        self.assertEqual(added, 10)
        self.assert_sequential_transitions(list(buffer.buffer), 100.0, 200.0)

    def test_partial_replay_keeps_internal_next_states(self):
        buffer = offline_replay.OfflineReplayBuffer(capacity=4)
        added = offline_replay.add_block_transitions(
            buffer,
            self.state,
            self.next_state,
            100.0,
            self.offsets,
            self.rewards,
            next_hoprate=200.0,
            max_transitions=4,
        )

        self.assertEqual(added, 4)
        transitions = list(buffer.buffer)
        for block_idx, transition in enumerate(transitions):
            np.testing.assert_array_equal(transition[5], self.state)
            self.assertEqual(transition[6], 100.0)
            self.assertEqual(int(transition[7]), block_idx + 1)
            self.assertFalse(transition[8])

    def test_special_helper_matches_sequential_semantics(self):
        buffer = SpecialReplayBuffer(capacity=10)
        add_special_block_transitions(
            buffer,
            self.state,
            self.next_state,
            100.0,
            self.offsets,
            self.rewards,
        )

        self.assert_sequential_transitions(list(buffer.buffer), 100.0, 100.0)


class OfflineReplayVersionTests(unittest.TestCase):
    def setUp(self):
        self.source = offline_replay.OfflineReplayBuffer(capacity=10)
        offline_replay.add_block_transitions(
            self.source,
            np.ones((2, 3), dtype=np.float32),
            np.full((2, 3), 2.0, dtype=np.float32),
            100.0,
            np.arange(10, dtype=np.int64),
            np.arange(10, dtype=np.float32),
            next_hoprate=200.0,
        )

    def test_v2_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replay_v2.npz"
            offline_replay.save_replay_buffer(path, self.source)
            destination = offline_replay.OfflineReplayBuffer(capacity=10)

            count, metadata = offline_replay.load_replay_into_buffer(
                path,
                destination,
                expected_observation_shape=(2, 3),
                expected_num_actions=20,
            )

        self.assertEqual(count, 10)
        self.assertEqual(metadata["format_version"], 2)
        for source_transition, loaded_transition in zip(
            self.source.buffer,
            destination.buffer,
        ):
            np.testing.assert_array_equal(
                source_transition[0],
                loaded_transition[0],
            )
            np.testing.assert_array_equal(
                source_transition[5],
                loaded_transition[5],
            )
            self.assertEqual(source_transition[1:5], loaded_transition[1:5])
            self.assertEqual(source_transition[6:], loaded_transition[6:])

    def _write_replay_with_metadata(self, path, metadata):
        arrays = offline_replay._buffer_to_arrays(self.source)
        np.savez_compressed(
            path,
            **arrays,
            metadata=np.asarray(json.dumps(metadata)),
        )

    def test_v1_replay_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replay_v1.npz"
            np.savez_compressed(
                path,
                metadata=np.asarray(json.dumps({"format_version": 1})),
            )
            with self.assertRaisesRegex(ValueError, "expected 2"):
                offline_replay.load_replay_into_buffer(
                    path,
                    offline_replay.OfflineReplayBuffer(capacity=10),
                )

    def test_replay_without_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replay_without_version.npz"
            self._write_replay_with_metadata(path, {})
            with self.assertRaisesRegex(ValueError, "missing format_version"):
                offline_replay.load_replay_into_buffer(
                    path,
                    offline_replay.OfflineReplayBuffer(capacity=10),
                )

    def test_unknown_replay_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replay_unknown_version.npz"
            np.savez_compressed(
                path,
                metadata=np.asarray(json.dumps({"format_version": 99})),
            )
            with self.assertRaisesRegex(ValueError, "expected 2"):
                offline_replay.load_replay_into_buffer(
                    path,
                    offline_replay.OfflineReplayBuffer(capacity=10),
                )


class TargetCriticInitializationTests(unittest.TestCase):
    SAC_CLASSES = (MainSAC, SpecialSAC)

    @staticmethod
    def make_agent(sac_class):
        return sac_class(
            n_actions=4,
            actor_lr=1e-4,
            critic_lr=1e-4,
            alpha_lr=1e-4,
            target_entropy=0.5,
            tau=0.01,
            gamma=0.95,
            device=torch.device("cpu"),
        )

    @staticmethod
    def assert_state_dicts_equal(test_case, online, target):
        online_state = online.state_dict()
        target_state = target.state_dict()
        test_case.assertEqual(online_state.keys(), target_state.keys())
        for key, value in online_state.items():
            test_case.assertTrue(
                torch.equal(value, target_state[key]),
                msg=f"State differs for {key}",
            )

    def test_first_target_calculation_initializes_exact_copies_once(self):
        for sac_class in self.SAC_CLASSES:
            with self.subTest(sac_class=sac_class.__module__):
                agent = self.make_agent(sac_class)
                next_imgs = torch.randn(2, 1, 8, 8)
                next_extras = torch.tensor([[100.0, 1.0], [100.0, 2.0]])
                rewards = torch.zeros(2, 1)
                dones = torch.zeros(2, 1)

                self.assertFalse(agent._target_critics_initialized)
                agent.calc_target(rewards, next_imgs, next_extras, dones)

                self.assertTrue(agent._target_critics_initialized)
                self.assertFalse(agent.target_critic_1.training)
                self.assertFalse(agent.target_critic_2.training)
                for network in (
                    agent.critic_1,
                    agent.critic_2,
                    agent.target_critic_1,
                    agent.target_critic_2,
                ):
                    self.assertFalse(
                        any(
                            isinstance(parameter, UninitializedParameter)
                            for parameter in network.parameters()
                        )
                    )
                self.assert_state_dicts_equal(
                    self,
                    agent.critic_1,
                    agent.target_critic_1,
                )
                self.assert_state_dicts_equal(
                    self,
                    agent.critic_2,
                    agent.target_critic_2,
                )

                target_snapshot = {
                    key: value.clone()
                    for key, value in agent.target_critic_1.state_dict().items()
                }
                with torch.no_grad():
                    next(agent.critic_1.parameters()).add_(1.0)
                agent.calc_target(rewards, next_imgs, next_extras, dones)
                for key, value in agent.target_critic_1.state_dict().items():
                    self.assertTrue(torch.equal(value, target_snapshot[key]))

    def test_update_after_lazy_initialization_has_finite_losses(self):
        rng = np.random.default_rng(7)
        transition_dict = {
            "state_imgs": rng.normal(size=(4, 8, 8)).astype(np.float32),
            "hoprates": np.full(4, 100.0, dtype=np.float32),
            "block_idxs": np.arange(4, dtype=np.float32),
            "actions": np.arange(4, dtype=np.int64),
            "rewards": rng.normal(size=4).astype(np.float32),
            "next_state_imgs": rng.normal(size=(4, 8, 8)).astype(np.float32),
            "next_hoprates": np.full(4, 100.0, dtype=np.float32),
            "next_block_idxs": np.arange(1, 5, dtype=np.float32),
            "dones": np.zeros(4, dtype=np.float32),
        }

        for sac_class in self.SAC_CLASSES:
            with self.subTest(sac_class=sac_class.__module__):
                stats = self.make_agent(sac_class).update(transition_dict)
                for key, value in stats.items():
                    self.assertTrue(np.isfinite(value), msg=f"{key} is not finite")


if __name__ == "__main__":
    unittest.main()
