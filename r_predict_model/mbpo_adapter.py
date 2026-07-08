"""Adapters between the FHSS discrete SAC replay format and MBPO reward model."""

import numpy as np


def encode_transition_inputs(state_imgs, hoprates, action_arrs, actions):
    """Flatten SAC transition fields into reward-model inputs."""
    state_flat = np.asarray(state_imgs, dtype=np.float32).reshape(len(state_imgs), -1)
    hoprates = np.asarray(hoprates, dtype=np.float32).reshape(-1, 1)
    action_arrs = np.asarray(action_arrs, dtype=np.float32).reshape(len(state_imgs), -1)
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([state_flat, hoprates, action_arrs, actions], axis=1)


def replay_sample_to_model_data(sample):
    inputs = encode_transition_inputs(
        sample["state_imgs"],
        sample["hoprates"],
        sample["action_arrs"],
        sample["actions"],
    )
    labels = np.asarray(sample["rewards"], dtype=np.float32).reshape(-1, 1)
    return inputs, labels


def infer_next_action_arr(action_arr, sampled_next_action_arr, sampled_action, new_action):
    """
    Update one slot in the action history.

    The baseline replay format does not store the sub-decision index, and action
    0 is also the zero-fill value.  Prefer the real transition's diff when it is
    visible; otherwise use the first zero slot as the same approximation the
    existing history encoding already relies on.
    """
    action_arr = np.asarray(action_arr, dtype=np.float32).copy()
    sampled_next_action_arr = np.asarray(sampled_next_action_arr, dtype=np.float32)
    diff_idx = np.flatnonzero(np.abs(sampled_next_action_arr - action_arr) > 1e-6)
    if len(diff_idx) > 0:
        fill_idx = int(diff_idx[0])
    else:
        zero_idx = np.flatnonzero(np.abs(action_arr) < 1e-6)
        fill_idx = int(zero_idx[0]) if len(zero_idx) > 0 else len(action_arr) - 1
    action_arr[fill_idx] = float(new_action)
    return action_arr


def concat_transition_batches(batches):
    """Concatenate non-empty SAC replay sample dictionaries."""
    valid_batches = [batch for batch in batches if batch is not None]
    if not valid_batches:
        raise ValueError("Need at least one batch to concatenate.")
    keys = valid_batches[0].keys()
    return {
        key: np.concatenate([batch[key] for batch in valid_batches], axis=0)
        for key in keys
    }


def train_reward_model_from_replay(reward_model, replay_buffer, batch_size):
    sample = replay_buffer.sample(replay_buffer.size())
    inputs, labels = replay_sample_to_model_data(sample)
    return reward_model.train(inputs, labels, batch_size=batch_size)


def rollout_reward_model(
    reward_model,
    agent,
    real_buffer,
    model_buffer,
    batch_size,
    fixed_hoprate,
    n_actions,
    deterministic_model=False,
):
    starts = real_buffer.sample(batch_size)
    state_imgs = starts["state_imgs"]
    hoprates = np.full(len(state_imgs), float(fixed_hoprate), dtype=np.float32)
    action_arrs = starts["action_arrs"]

    actions = np.zeros(len(state_imgs), dtype=np.int64)
    for i in range(len(state_imgs)):
        action = agent.take_action(state_imgs[i], float(fixed_hoprate), action_arrs[i])
        actions[i] = int(np.clip(action, 0, n_actions - 1))

    model_inputs = encode_transition_inputs(state_imgs, hoprates, action_arrs, actions)
    rewards = reward_model.predict_reward(model_inputs, deterministic=deterministic_model)

    for i in range(len(state_imgs)):
        next_action_arr = infer_next_action_arr(
            action_arrs[i],
            starts["next_action_arrs"][i],
            starts["actions"][i],
            actions[i],
        )
        model_buffer.add(
            state_imgs[i],
            float(fixed_hoprate),
            action_arrs[i],
            int(actions[i]),
            float(rewards[i]),
            state_imgs[i].copy(),
            float(fixed_hoprate),
            next_action_arr,
            False,
        )

    return {
        "generated": int(len(state_imgs)),
        "reward_mean": float(np.mean(rewards)) if len(rewards) else 0.0,
        "reward_std": float(np.std(rewards)) if len(rewards) else 0.0,
    }
