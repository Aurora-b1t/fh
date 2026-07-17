"""
Baseline training entry point for FHSS anti-jamming reinforcement learning.

Builds the FHSSQPSKEnv environment, baseline discrete SAC agent, and replay
buffer, then runs the autoregressive 10-offset training loop with logging and
reward/BER/loss plots.
"""

import argparse
import os
import time
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt

from fh_env import FHSSQPSKEnv
from SAC import SAC, ReplayBuffer
from offline_replay import (
    add_block_transitions,
    environment_metadata,
    load_replay_into_buffer,
)
import settings

def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True,
    )
    return logging.getLogger()

def build_agent_and_env(args):
    # -------------------------------------------------------------------------
    # 1. Device Configuration (GPU/CPU)
    # -------------------------------------------------------------------------
    if torch.cuda.is_available() and not args.cpu_only:
        device = torch.device("cuda")
        logging.info(f"Training Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logging.info("Training Device: CPU")

    # -------------------------------------------------------------------------
    # 2. Environment Initialization
    # -------------------------------------------------------------------------
    # Pass configuration from settings
    env = FHSSQPSKEnv(**settings.ENV_CONFIG)

    # Number of discrete actions matches number of channels
    n_actions = env.num_channels
    logging.info(f"Environment Initialized. Num Channels/Actions: {n_actions}")

    # -------------------------------------------------------------------------
    # 3. Build SAC Agent
    # -------------------------------------------------------------------------
    # Target entropy is typically -dim(A) for continuous, or relative to log(|A|) for discrete
    target_entropy = np.log(n_actions) * settings.SAC_CONFIG["target_entropy_ratio"]
    
    agent = SAC(
        n_actions=n_actions,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        target_entropy=target_entropy,
        tau=args.tau,
        gamma=args.gamma,
        device=device,
    )

    # -------------------------------------------------------------------------
    # 4. Replay Buffer
    # -------------------------------------------------------------------------
    buffer = ReplayBuffer(capacity=args.replay_size)

    return env, agent, buffer, device, n_actions


def compute_block_rewards(ber_blocks, hoprate):
    """
    Calculate rewards for each of the 10 blocks in a step.
    Keep the same scale as FHSSQPSKEnv.step(): base - BER penalty - hoprate penalty.
    """
    base = settings.REWARD_CONFIG["base_reward"]
    ber_p = settings.REWARD_CONFIG["ber_penalty"]
    hop_p = settings.REWARD_CONFIG["hoprate_penalty"]
    rewards = [
        base - ber_p * float(ber) - hop_p * float(hoprate)
        for ber in ber_blocks
    ]
    return rewards


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    logger = setup_logger(log_path)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_path}")

    # Build components
    env, agent, buffer, device, n_actions = build_agent_and_env(args)

    state_img, info = env.reset()
    loaded_count, replay_metadata = load_replay_into_buffer(
        args.offline_replay_path,
        buffer,
        expected_observation_shape=np.asarray(state_img).shape,
        expected_num_actions=n_actions,
        current_environment_metadata=environment_metadata(
            settings.ENV_CONFIG,
            settings.JAMMER_CONFIG,
            settings.REWARD_CONFIG,
        ),
        logger=logging.getLogger(),
    )
    logger.info(
        "Loaded %d offline real transitions from %s (mode=%s)",
        loaded_count,
        args.offline_replay_path,
        replay_metadata.get("hoprate_mode", "unknown"),
    )

    # Use fixed hoprate for online training, matching the existing experiment.
    fixed_hoprate = settings.TRAIN_CONFIG["fixed_hoprate"]

    logger.info(f"Start Training for 1 episode with {args.steps_per_episode} steps...")
    logger.info(f"Batch Size: {args.batch_size}, Updates per step: {args.update_iters_per_step}")
    
    start_time = time.time()
    total_steps = 0
    episode = 1
    
    ep_start_time = time.time()
    
    ep_block_rewards = []
    
    # Tracking for plots
    plot_rewards = []
    plot_losses_actor = []
    plot_losses_critic = []
    plot_bers = []
    
    logger.info(f"--- Episode {episode} Start ---")

    # Main Loop
    for step_idx in range(1, args.steps_per_episode + 1):
        step_start_time = time.time()
        
        # -------------------------------------------------------
        # 1. Decision Making for 10 block offsets
        # -------------------------------------------------------
        offsets = np.zeros(10, dtype=np.float32)

        for i in range(10):
            # Take action using current state and block position.
            a_i = agent.take_action(state_img, fixed_hoprate, i)
            a_i = int(np.clip(a_i, 0, n_actions - 1))
            offsets[i] = a_i

        # -------------------------------------------------------
        # 2. Environment Step
        # -------------------------------------------------------
        # Execute the sequence of 10 offsets
        next_state_img, _reward_total, terminated, truncated, info = env.step(
            {"hoprate": fixed_hoprate, "offsets": offsets}
        )

        # -------------------------------------------------------
        # 3. Reward Calculation & Storage
        # -------------------------------------------------------
        ber_blocks = info.get("ber_blocks", [])
        
        # Compute rewards for each block individually
        per_block_rewards = compute_block_rewards(ber_blocks, info.get("hoprate_used", fixed_hoprate))
        ep_block_rewards.extend(per_block_rewards)
        
        mean_step_ber = np.mean(ber_blocks) if len(ber_blocks) > 0 else 0.0
        mean_step_reward = np.mean(per_block_rewards) if len(per_block_rewards) > 0 else 0.0

        add_block_transitions(
            buffer,
            state_img,
            next_state_img,
            info.get("hoprate_used", fixed_hoprate),
            offsets,
            per_block_rewards,
            next_hoprate=fixed_hoprate,
        )

        # Move to next state
        state_img = next_state_img
        total_steps += 1

        # -------------------------------------------------------
        # 4. Training Update
        # -------------------------------------------------------
        train_stats = {}
        for _ in range(args.update_iters_per_step):
            batch = buffer.sample(args.batch_size)
            stats = agent.update(batch)
            train_stats = stats

        step_duration = time.time() - step_start_time
        
        # Logging Data
        plot_rewards.append(mean_step_reward)
        plot_bers.append(mean_step_ber)
        plot_losses_actor.append(train_stats.get('actor_loss', 0) if train_stats else 0)
        plot_losses_critic.append(train_stats.get('critic1_loss', 0) if train_stats else 0)

        log_msg = (f"Step {step_idx}/{args.steps_per_episode} | "
                   f"Offsets: {offsets.astype(int).tolist()} | "
                   f"Rew: {mean_step_reward:.4f} | "
                   f"BER: {mean_step_ber:.4f}")
        
        if train_stats:
             log_msg += (f" | Loss: A={train_stats.get('actor_loss', 0):.3f}, "
                         f"C={train_stats.get('critic1_loss', 0):.3f}, "
                         f"Alpha={train_stats.get('alpha', 0):.5f}")
        
        log_msg += f" | T: {step_duration:.2f}s"
        logger.info(log_msg)

        if terminated or truncated:
            logger.info("Episode terminated early.")
            break

    # -------------------------------------------------------
    # End of Episode
    # -------------------------------------------------------
    ep_duration = time.time() - ep_start_time
    mean_ep_reward = float(np.mean(ep_block_rewards)) if len(ep_block_rewards) > 0 else 0.0
    
    logger.info(f"--- Episode {episode} Finished ---")

    # Plotting
    try:
        # 1. Reward
        plt.figure()
        plt.plot(plot_rewards)
        plt.title("Mean Step Reward")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "reward.png"))
        plt.close()

        # 2. BER
        plt.figure()
        plt.plot(plot_bers, color='r')
        plt.title("Mean Step BER")
        plt.xlabel("Step")
        plt.ylabel("BER")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "ber.png"))
        plt.close()

        # 3. Loss (Auto-scaled)
        plt.figure()
        plt.plot(plot_losses_actor, label="Actor Loss", alpha=0.7)
        plt.plot(plot_losses_critic, label="Critic Loss", alpha=0.7)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)

        # Scale Y-axis to ignore initial spikes
        skip = max(5, int(len(plot_losses_critic) * 0.05))
        if len(plot_losses_critic) > skip:
            valid_vals = plot_losses_actor[skip:] + plot_losses_critic[skip:]
            if valid_vals:
                y_min, y_max = np.percentile(valid_vals, [1, 99])
                yr = y_max - y_min if y_max != y_min else 1.0
                plt.ylim(y_min - yr * 0.1, y_max + yr * 0.1)

        plt.savefig(os.path.join(args.output_dir, "loss.png"))
        plt.close()
        logger.info(f"Plots saved to {args.output_dir}.")
        
    except Exception as e:
        logger.error(f"Plotting failed: {e}")

    total_duration = time.time() - start_time
    logger.info(f"Total Time: {total_duration:.2f}s | Mean Ep Reward: {mean_ep_reward:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_episode", type=int, default=settings.TRAIN_CONFIG["steps_per_episode"])
    parser.add_argument("--output_dir", type=str, default="outputs/offsets/pre50000/comb/512_start")
    parser.add_argument("--log_file", type=str, default="training_log.txt")

    # Agent Params
    parser.add_argument("--actor_lr", type=float, default=settings.SAC_CONFIG["actor_lr"])
    parser.add_argument("--critic_lr", type=float, default=settings.SAC_CONFIG["critic_lr"])
    parser.add_argument("--alpha_lr", type=float, default=settings.SAC_CONFIG["alpha_lr"])
    parser.add_argument("--tau", type=float, default=settings.SAC_CONFIG["tau"])
    parser.add_argument("--gamma", type=float, default=settings.SAC_CONFIG["gamma"])

    # Buffer Params
    parser.add_argument("--replay_size", type=int, default=settings.BUFFER_CONFIG["capacity"])
    parser.add_argument("--batch_size", type=int, default=settings.BUFFER_CONFIG["batch_size"])
    parser.add_argument("--update_iters_per_step", type=int, default=settings.TRAIN_CONFIG["update_iters_per_step"])

    parser.add_argument(
        "--offline_replay_path",
        type=str,
        default=settings.OFFLINE_REPLAY_CONFIG["default_path"],
        help="Offline real replay .npz file loaded before the first update.",
    )

    parser.add_argument("--cpu_only", action="store_true", default=settings.CPU_ONLY)
    return parser.parse_args()


if __name__ == "__main__":
    settings.set_random_seeds()
    args = parse_args()
    train(args)
