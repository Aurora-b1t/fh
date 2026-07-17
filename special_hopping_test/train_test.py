"""Online SAC training entry point for the isolated hopping-pattern test."""

import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from fh_env_test import FHSSQPSKEnv
from SAC_test import ReplayBuffer, SAC
import test_settings as settings


NUM_BLOCKS = 10


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger()


def build_agent_and_env():
    if torch.cuda.is_available() and not settings.CPU_ONLY:
        device = torch.device("cuda")
        logging.info("Training Device: GPU (%s)", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logging.info("Training Device: CPU")

    env = FHSSQPSKEnv(**settings.ENV_CONFIG)
    n_actions = env.num_channels
    if n_actions != 20:
        raise ValueError(f"Special test requires 20 actions, got {n_actions}.")
    logging.info("Environment Initialized. Num Channels/Actions: %d", n_actions)

    target_entropy = (
        np.log(n_actions) * settings.SAC_CONFIG["target_entropy_ratio"]
    )
    agent = SAC(
        n_actions=n_actions,
        actor_lr=settings.SAC_CONFIG["actor_lr"],
        critic_lr=settings.SAC_CONFIG["critic_lr"],
        alpha_lr=settings.SAC_CONFIG["alpha_lr"],
        target_entropy=target_entropy,
        tau=settings.SAC_CONFIG["tau"],
        gamma=settings.SAC_CONFIG["gamma"],
        device=device,
    )
    buffer = ReplayBuffer(capacity=settings.BUFFER_CONFIG["capacity"])
    return env, agent, buffer, n_actions


def compute_block_rewards(ber_blocks, hoprate):
    base = settings.REWARD_CONFIG["base_reward"]
    ber_penalty = settings.REWARD_CONFIG["ber_penalty"]
    hoprate_penalty = settings.REWARD_CONFIG["hoprate_penalty"]
    return [
        base
        - ber_penalty * float(ber)
        - hoprate_penalty * float(hoprate)
        for ber in ber_blocks
    ]


def add_block_transitions(
    buffer,
    state_img,
    next_state_img,
    hoprate,
    offsets,
    per_block_rewards,
):
    """Store one environment step as ten block-level transitions."""
    if len(offsets) != NUM_BLOCKS or len(per_block_rewards) != NUM_BLOCKS:
        raise ValueError("Exactly ten offsets and block rewards are required.")

    for block_idx in range(NUM_BLOCKS):
        is_last_block = block_idx == NUM_BLOCKS - 1
        buffer.add(
            state_img,
            hoprate,
            block_idx,
            int(offsets[block_idx]),
            float(per_block_rewards[block_idx]),
            next_state_img if is_last_block else state_img,
            hoprate,
            (block_idx + 1) % NUM_BLOCKS,
            False,
        )


def save_training_plots(output_dir, rewards, bers, actor_losses, critic_losses):
    plt.figure()
    plt.plot(rewards)
    plt.title("Mean Step Reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "reward.png"))
    plt.close()

    plt.figure()
    plt.plot(bers, color="r")
    plt.title("Mean Step BER")
    plt.xlabel("Step")
    plt.ylabel("BER")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "ber.png"))
    plt.close()

    plt.figure()
    plt.plot(actor_losses, label="Actor Loss", alpha=0.7)
    plt.plot(critic_losses, label="Critic Loss", alpha=0.7)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)

    skip = max(5, int(len(critic_losses) * 0.05))
    if len(critic_losses) > skip:
        valid_vals = actor_losses[skip:] + critic_losses[skip:]
        if valid_vals:
            y_min, y_max = np.percentile(valid_vals, [1, 99])
            y_range = y_max - y_min if y_max != y_min else 1.0
            plt.ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()


def train(args):
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(output_dir, "training_log.txt"))

    capture_steps = sorted({int(step) for step in args.capture_steps})
    if any(step <= 0 for step in capture_steps):
        raise ValueError("capture_steps must contain positive one-based steps.")

    logger.info("Output directory: %s", output_dir)
    logger.info("PSD capture steps: %s", capture_steps)

    env, agent, buffer, n_actions = build_agent_and_env()
    env.configure_psd_capture(capture_steps, output_dir)
    state_img, _ = env.reset()

    fixed_hoprate = settings.TRAIN_CONFIG["fixed_hoprate"]
    batch_size = settings.BUFFER_CONFIG["batch_size"]
    update_iters = settings.TRAIN_CONFIG["update_iters_per_step"]

    logger.info(
        "Start online training for 1 episode with %d steps.",
        args.steps_per_episode,
    )
    logger.info(
        "Gamma: %.2f | Batch Size: %d | Updates per step: %d",
        settings.SAC_CONFIG["gamma"],
        batch_size,
        update_iters,
    )
    logger.info("Offline replay: disabled | MBPO: disabled")
    logger.info("--- Episode 1 Start ---")

    start_time = time.time()
    episode_block_rewards = []
    plot_rewards = []
    plot_bers = []
    plot_actor_losses = []
    plot_critic_losses = []

    for step_idx in range(1, args.steps_per_episode + 1):
        step_start_time = time.time()
        offsets = np.zeros(NUM_BLOCKS, dtype=np.float32)

        for block_idx in range(NUM_BLOCKS):
            action = agent.take_action(state_img, fixed_hoprate, block_idx)
            offsets[block_idx] = int(np.clip(action, 0, n_actions - 1))

        next_state_img, _, terminated, truncated, info = env.step(
            {"hoprate": fixed_hoprate, "offsets": offsets}
        )

        ber_blocks = info.get("ber_blocks", [])
        if len(ber_blocks) != NUM_BLOCKS:
            raise RuntimeError(
                f"Environment returned {len(ber_blocks)} BER values; expected 10."
            )

        hoprate_used = info.get("hoprate_used", fixed_hoprate)
        per_block_rewards = compute_block_rewards(ber_blocks, hoprate_used)
        episode_block_rewards.extend(per_block_rewards)

        mean_step_ber = float(np.mean(ber_blocks))
        mean_step_reward = float(np.mean(per_block_rewards))

        add_block_transitions(
            buffer,
            state_img,
            next_state_img,
            hoprate_used,
            offsets,
            per_block_rewards,
        )
        state_img = next_state_img

        train_stats = {}
        if buffer.size() >= batch_size:
            for _ in range(update_iters):
                train_stats = agent.update(buffer.sample(batch_size))

        plot_rewards.append(mean_step_reward)
        plot_bers.append(mean_step_ber)
        plot_actor_losses.append(train_stats.get("actor_loss", 0.0))
        plot_critic_losses.append(train_stats.get("critic1_loss", 0.0))

        phase = info.get("comb_phase")
        expected = (
            settings.EXPECTED_OFFSETS[phase]
            if phase in (0, 1)
            else None
        )
        log_msg = (
            f"Step {step_idx}/{args.steps_per_episode} | Phase: {phase} | "
            f"Offsets: {offsets.astype(int).tolist()} | "
            f"Expected: {expected} | Rew: {mean_step_reward:.4f} | "
            f"BER: {mean_step_ber:.4f} | Buffer: {buffer.size()}"
        )
        if train_stats:
            log_msg += (
                f" | Loss: A={train_stats.get('actor_loss', 0):.3f}, "
                f"C={train_stats.get('critic1_loss', 0):.3f}, "
                f"Alpha={train_stats.get('alpha', 0):.5f}"
            )
        else:
            log_msg += f" | Warmup: {buffer.size()}/{batch_size}"
        log_msg += f" | T: {time.time() - step_start_time:.2f}s"
        logger.info(log_msg)

        if terminated or truncated:
            logger.info("Episode terminated early.")
            break

    logger.info("--- Episode 1 Finished ---")
    save_training_plots(
        output_dir,
        plot_rewards,
        plot_bers,
        plot_actor_losses,
        plot_critic_losses,
    )

    mean_episode_reward = (
        float(np.mean(episode_block_rewards))
        if episode_block_rewards
        else 0.0
    )
    logger.info("Plots saved to %s.", output_dir)
    logger.info(
        "Total Time: %.2fs | Mean Ep Reward: %.4f",
        time.time() - start_time,
        mean_episode_reward,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone online SAC test for the special hopping pattern."
    )
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=settings.TRAIN_CONFIG["steps_per_episode"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=settings.DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--capture_steps",
        type=int,
        nargs="*",
        default=settings.CAPTURE_STEPS,
        help="One-based steps whose input obs and ten block PSDs are saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    settings.set_random_seeds()
    train(parse_args())
