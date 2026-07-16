"""Shared offline replay serialization and transition helpers."""

import json
import logging
import os
from collections import deque

import numpy as np


FORMAT_VERSION = 1
NUM_BLOCKS = 10
REPLAY_KEYS = (
    "state_imgs",
    "hoprates",
    "block_idxs",
    "actions",
    "rewards",
    "next_state_imgs",
    "next_hoprates",
    "next_block_idxs",
    "dones",
)


class OfflineReplayBuffer:
    """Small replay-compatible collector used by the standalone generator."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state_img,
        hoprate,
        block_idx,
        action,
        reward,
        next_state_img,
        next_hoprate,
        next_block_idx,
        done,
    ):
        self.buffer.append(
            (
                np.asarray(state_img, dtype=np.float32),
                float(hoprate),
                float(block_idx),
                int(action),
                float(reward),
                np.asarray(next_state_img, dtype=np.float32),
                float(next_hoprate),
                float(next_block_idx),
                bool(done),
            )
        )

    def size(self):
        return len(self.buffer)


def compute_block_rewards(ber_blocks, hoprate, reward_config):
    """Convert the environment's per-block BER values into replay rewards."""
    base = reward_config["base_reward"]
    ber_penalty = reward_config["ber_penalty"]
    hoprate_penalty = reward_config["hoprate_penalty"]
    return [
        base - ber_penalty * float(ber) - hoprate_penalty * float(hoprate)
        for ber in ber_blocks
    ]


def add_block_transitions(
    buffer,
    state_img,
    next_state_img,
    hoprate,
    offsets,
    per_block_rewards,
    next_hoprate=None,
    max_transitions=None,
):
    """Add one FHSS environment step as up to ten replay transitions."""
    if next_hoprate is None:
        next_hoprate = hoprate

    added = 0
    for block_idx in range(NUM_BLOCKS):
        if max_transitions is not None and added >= max_transitions:
            break
        buffer.add(
            state_img,
            hoprate,
            block_idx,
            int(offsets[block_idx]),
            float(per_block_rewards[block_idx]),
            next_state_img,
            next_hoprate,
            (block_idx + 1) % NUM_BLOCKS,
            False,
        )
        added += 1
    return added


def _as_jsonable(value):
    """Convert NumPy scalar/container values before writing metadata."""
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def environment_metadata(env_config, jammer_config, reward_config):
    """Return the configuration snapshot stored with generated replay data."""
    return {
        "env_config": _as_jsonable(env_config),
        "jammer_config": _as_jsonable(jammer_config),
        "reward_config": _as_jsonable(reward_config),
    }


def _buffer_to_arrays(buffer):
    transitions = list(buffer.buffer)
    if not transitions:
        raise ValueError("Cannot save an empty replay buffer.")
    columns = list(zip(*transitions))
    return {
        "state_imgs": np.asarray(columns[0], dtype=np.float32),
        "hoprates": np.asarray(columns[1], dtype=np.float32),
        "block_idxs": np.asarray(columns[2], dtype=np.float32),
        "actions": np.asarray(columns[3], dtype=np.int64),
        "rewards": np.asarray(columns[4], dtype=np.float32),
        "next_state_imgs": np.asarray(columns[5], dtype=np.float32),
        "next_hoprates": np.asarray(columns[6], dtype=np.float32),
        "next_block_idxs": np.asarray(columns[7], dtype=np.float32),
        "dones": np.asarray(columns[8], dtype=np.float32),
    }


def save_replay_buffer(path, buffer, metadata=None):
    """Save a ReplayBuffer in a portable compressed NumPy format."""
    arrays = _buffer_to_arrays(buffer)
    metadata = dict(metadata or {})
    metadata.update(
        {
            "format_version": FORMAT_VERSION,
            "num_transitions": int(len(buffer.buffer)),
            "observation_shape": list(arrays["state_imgs"].shape[1:]),
            "num_actions_observed": int(np.max(arrays["actions"])) + 1,
            "num_blocks": NUM_BLOCKS,
        }
    )
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        metadata=np.asarray(json.dumps(_as_jsonable(metadata), sort_keys=True)),
    )


def _load_arrays(path):
    try:
        archive = np.load(path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Offline replay file does not exist: {path}") from exc
    except Exception as exc:
        raise ValueError(f"Could not read offline replay file '{path}': {exc}") from exc

    with archive:
        missing = [key for key in REPLAY_KEYS if key not in archive]
        if missing:
            raise ValueError(f"Offline replay is missing fields: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in REPLAY_KEYS}
        metadata_value = archive["metadata"] if "metadata" in archive else np.asarray("{}")
        try:
            metadata = json.loads(str(metadata_value.item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Offline replay metadata is not valid JSON.") from exc
    return arrays, metadata


def load_replay_into_buffer(
    path,
    buffer,
    expected_observation_shape=None,
    expected_num_actions=None,
    current_environment_metadata=None,
    logger=None,
):
    """Validate an offline replay file and append it to a ReplayBuffer."""
    arrays, metadata = _load_arrays(path)
    file_version = metadata.get("format_version")
    if file_version is not None and int(file_version) != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported offline replay format version: {file_version}; "
            f"expected {FORMAT_VERSION}."
        )
    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Offline replay fields have inconsistent lengths: {lengths}")
    count = next(iter(lengths.values()))
    if count == 0:
        raise ValueError("Offline replay contains no transitions.")

    state_shape = tuple(arrays["state_imgs"].shape[1:])
    next_state_shape = tuple(arrays["next_state_imgs"].shape[1:])
    if state_shape != next_state_shape:
        raise ValueError(
            f"State and next-state shapes differ: {state_shape} vs {next_state_shape}"
        )
    if expected_observation_shape is not None and state_shape != tuple(expected_observation_shape):
        raise ValueError(
            "Offline replay observation shape does not match the current environment: "
            f"file={state_shape}, environment={tuple(expected_observation_shape)}"
        )
    if expected_num_actions is not None:
        actions = arrays["actions"]
        if np.any(actions < 0) or np.any(actions >= expected_num_actions):
            raise ValueError(
                f"Offline replay contains actions outside [0, {expected_num_actions - 1}]."
            )
        file_num_actions = metadata.get("num_actions")
        if file_num_actions is not None and int(file_num_actions) != int(expected_num_actions):
            raise ValueError(
                "Offline replay action-space size does not match the current environment: "
                f"file={file_num_actions}, environment={expected_num_actions}"
            )
    for key in ("block_idxs", "next_block_idxs"):
        if np.any(arrays[key] < 0) or np.any(arrays[key] >= NUM_BLOCKS):
            raise ValueError(f"Offline replay contains invalid {key} values.")
    if int(metadata.get("num_blocks", NUM_BLOCKS)) != NUM_BLOCKS:
        raise ValueError("Offline replay was generated with a different block count.")
    capacity = buffer.buffer.maxlen
    if capacity is not None and count > capacity:
        raise ValueError(
            f"Offline replay has {count} transitions but replay capacity is only {capacity}."
        )

    for idx in range(count):
        buffer.add(
            arrays["state_imgs"][idx],
            arrays["hoprates"][idx],
            arrays["block_idxs"][idx],
            arrays["actions"][idx],
            arrays["rewards"][idx],
            arrays["next_state_imgs"][idx],
            arrays["next_hoprates"][idx],
            arrays["next_block_idxs"][idx],
            arrays["dones"][idx],
        )

    if current_environment_metadata is not None:
        stored_config = {
            key: metadata.get(key)
            for key in ("env_config", "jammer_config", "reward_config")
        }
        current_config = _as_jsonable(current_environment_metadata)
        if any(
            stored_config.get(key) is not None
            and current_config.get(key) != stored_config.get(key)
            for key in current_config
        ):
            message = (
                "Offline replay environment configuration differs from the current "
                "settings; use a data file generated for the intended environment."
            )
            (logger or logging.getLogger(__name__)).warning(message)

    return count, metadata
