"""Generate real-environment replay data for offset SAC training."""

import argparse
import logging

import numpy as np

from fh_env import FHSSQPSKEnv
import settings
from offline_replay import (
    add_block_transitions,
    compute_block_rewards,
    environment_metadata,
    OfflineReplayBuffer,
    save_replay_buffer,
)


def quantize_hoprate(hoprate, env):
    clipped = np.clip(hoprate, env.hoprate_min, env.hoprate_max)
    return float(int(round(clipped / 10.0)) * 10)


def make_hoprate_sampler(mode, fixed_hoprate, env, rng):
    if mode == "fixed":
        fixed = float(fixed_hoprate)
        quantized = quantize_hoprate(fixed, env)
        return lambda: quantized

    min_step = int(np.ceil(env.hoprate_min / 10.0))
    max_step = int(np.floor(env.hoprate_max / 10.0))
    if min_step > max_step:
        raise ValueError("Environment hoprate range does not contain a valid 10 Hz value.")
    valid_rates = np.arange(min_step, max_step + 1, dtype=np.int32) * 10
    return lambda: float(rng.choice(valid_rates))


def generate(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    settings.set_random_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    env = FHSSQPSKEnv(**settings.ENV_CONFIG)
    state_img, _info = env.reset()
    n_actions = env.num_channels
    buffer = OfflineReplayBuffer(capacity=args.num_transitions)
    hoprate_sampler = make_hoprate_sampler(args.hoprate_mode, args.fixed_hoprate, env, rng)

    current_hoprate = hoprate_sampler()
    env_steps = 0
    while buffer.size() < args.num_transitions:
        offsets = rng.integers(0, n_actions, size=10, dtype=np.int32).astype(np.float32)
        next_state_img, _reward, terminated, truncated, info = env.step(
            {"hoprate": current_hoprate, "offsets": offsets}
        )
        used_hoprate = float(info.get("hoprate_used", current_hoprate))
        next_hoprate = hoprate_sampler()
        ber_blocks = info.get("ber_blocks", [])
        per_block_rewards = compute_block_rewards(
            ber_blocks,
            used_hoprate,
            settings.REWARD_CONFIG,
        )
        remaining = args.num_transitions - buffer.size()
        add_block_transitions(
            buffer,
            state_img,
            next_state_img,
            used_hoprate,
            offsets,
            per_block_rewards,
            next_hoprate=next_hoprate,
            max_transitions=remaining,
        )
        state_img = next_state_img
        current_hoprate = next_hoprate
        env_steps += 1

        if terminated or truncated:
            state_img, _info = env.reset()
            current_hoprate = hoprate_sampler()

        if env_steps % 100 == 0 or buffer.size() == args.num_transitions:
            logging.info(
                "Collected %d/%d transitions (%d environment steps)",
                buffer.size(),
                args.num_transitions,
                env_steps,
            )

    metadata = environment_metadata(
        settings.ENV_CONFIG,
        settings.JAMMER_CONFIG,
        settings.REWARD_CONFIG,
    )
    metadata.update(
        {
            "generator": "generate_offline_replay.py",
            "seed": args.seed,
            "hoprate_mode": args.hoprate_mode,
            "fixed_hoprate": args.fixed_hoprate if args.hoprate_mode == "fixed" else None,
            "num_env_steps": env_steps,
            "num_actions": n_actions,
        }
    )
    save_replay_buffer(args.output_path, buffer, metadata)
    logging.info("Saved %d transitions to %s", buffer.size(), args.output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate real FHSS block transitions with a random policy."
    )
    parser.add_argument(
        "--num_transitions",
        type=int,
        default=settings.OFFLINE_REPLAY_CONFIG["num_transitions"],
        help="Number of block-level replay transitions to save.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=settings.OFFLINE_REPLAY_CONFIG["default_path"],
        help="Output .npz path.",
    )
    parser.add_argument(
        "--hoprate_mode",
        choices=("random", "fixed"),
        default=settings.OFFLINE_REPLAY_CONFIG["hoprate_mode"],
        help="Use a uniformly random valid hoprate or one fixed hoprate.",
    )
    parser.add_argument(
        "--fixed_hoprate",
        type=float,
        default=settings.TRAIN_CONFIG["fixed_hoprate"],
        help="Hoprate used when --hoprate_mode=fixed.",
    )
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
