"""Adapters between the FHSS discrete SAC replay format and MBPO reward model."""

import numpy as np


def encode_transition_inputs(state_imgs, hoprates, block_idxs, actions):
    """Flatten SAC transition fields into reward-model inputs."""
    state_flat = np.asarray(state_imgs, dtype=np.float32).reshape(len(state_imgs), -1)
    hoprates = np.asarray(hoprates, dtype=np.float32).reshape(-1, 1)
    block_idxs = np.asarray(block_idxs, dtype=np.float32).reshape(-1, 1)
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([state_flat, hoprates, block_idxs, actions], axis=1)


def replay_sample_to_model_data(sample):
    inputs = encode_transition_inputs(
        sample["state_imgs"],
        sample["hoprates"],
        sample["block_idxs"],
        sample["actions"],
    )
    labels = np.asarray(sample["rewards"], dtype=np.float32).reshape(-1, 1)
    return inputs, labels


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
    block_idxs = starts["block_idxs"]

    actions = np.zeros(len(state_imgs), dtype=np.int64)
    for i in range(len(state_imgs)):
        action = agent.take_action(state_imgs[i], float(fixed_hoprate), block_idxs[i])
        actions[i] = int(np.clip(action, 0, n_actions - 1))

    model_inputs = encode_transition_inputs(state_imgs, hoprates, block_idxs, actions)
    rewards = reward_model.predict_reward(model_inputs, deterministic=deterministic_model)

    for i in range(len(state_imgs)):
        block_idx = int(np.clip(round(float(block_idxs[i])), 0, 9))
        next_block_idx = min(block_idx + 1, 9)
        model_buffer.add(
            state_imgs[i],
            float(fixed_hoprate),
            block_idx,
            int(actions[i]),
            float(rewards[i]),
            state_imgs[i].copy(),
            float(fixed_hoprate),
            next_block_idx,
            False,
        )

    return {
        "generated": int(len(state_imgs)),
        "reward_mean": float(np.mean(rewards)) if len(rewards) else 0.0,
        "reward_std": float(np.std(rewards)) if len(rewards) else 0.0,
    }
