import argparse
import os
import time
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt

from fh_env_opt_newest import FHSSQPSKEnv
from SAC import ReplayBuffer, EASReplayBuffer
from SAC_eas_local import SACEASLocal
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
    if torch.cuda.is_available() and not args.cpu_only:
        device = torch.device("cuda")
        logging.info(f"Training Device: GPU ({torch.cuda.get_device_name(0)})")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        logging.info("Training Device: CPU")

    env = FHSSQPSKEnv(**settings.ENV_CONFIG)
    n_actions = env.num_channels
    logging.info(f"Environment Initialized. Num Channels/Actions: {n_actions}")

    target_entropy = np.log(n_actions) * settings.SAC_CONFIG["target_entropy_ratio"]

    agent = SACEASLocal(
        n_actions=n_actions,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        target_entropy=target_entropy,
        tau=args.tau,
        gamma=args.gamma,
        device=device,
        search_radius=args.search_radius,
        distill_coef=args.distill_coef,
        search_eval=args.search_eval,
        teacher_from_replay=args.teacher_from_replay,
        log_search_stats=args.log_search_stats,
        filter_teacher_on_update=args.filter_teacher_on_update,
        teacher_compare_mode=args.teacher_compare_mode,
    )

    buffer = ReplayBuffer(capacity=args.replay_size)
    eas_buffer = EASReplayBuffer(capacity=args.eas_replay_capacity)
    return env, agent, buffer, eas_buffer, device, n_actions


def compute_block_rewards(ber_blocks, hoprate):
    base = settings.REWARD_CONFIG["base_reward"]
    ber_p = settings.REWARD_CONFIG["ber_penalty"]
    hop_p = settings.REWARD_CONFIG["hoprate_penalty"]
    return [
        base - ber_p * float(ber) - hop_p * float(hoprate)
        for ber in ber_blocks
    ]


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    logger = setup_logger(log_path)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_path}")

    env, agent, buffer, eas_buffer, device, n_actions = build_agent_and_env(args)
    fixed_hoprate = settings.TRAIN_CONFIG["fixed_hoprate"]

    logger.info(f"Start EAS-local SAC Training for 1 episode with {args.steps_per_episode} steps...")
    logger.info(f"Batch Size: {args.batch_size}, Updates per step: {args.update_iters_per_step}")
    logger.info(f"Local search radius: {args.search_radius}, Distill coef: {args.distill_coef}")

    start_time = time.time()
    episode = 1
    ep_start_time = time.time()

    state_img, info = env.reset()
    ep_block_rewards = []

    plot_rewards = []
    plot_bers = []
    plot_losses_actor = []
    plot_losses_critic = []
    plot_losses_distill = []
    plot_search_change_rate = []
    plot_eas_valid_ratio = []

    logger.info(f"--- Episode {episode} Start ---")

    for step_idx in range(1, args.steps_per_episode + 1):
        step_start_time = time.time()

        offsets = np.zeros(10, dtype=np.float32)
        action_arr_before = np.zeros(10, dtype=np.float32)
        use_eas_rollout = buffer.size() >= args.min_buffer_before_train

        for i in range(10):
            if use_eas_rollout:
                a_i, search_info = agent.local_search_action(state_img, fixed_hoprate, action_arr_before)
            else:
                a_i = agent.take_action(state_img, fixed_hoprate, action_arr_before)
                search_info = None
            a_i = int(np.clip(a_i, 0, n_actions - 1))
            offsets[i] = a_i
            if use_eas_rollout and search_info["teacher_action"] != search_info["seed_action"]:
                eas_buffer.add(
                    state_img,
                    fixed_hoprate,
                    action_arr_before.copy(),
                    search_info["teacher_action"],
                    search_gain=search_info.get("search_gain", 0.0),
                    candidate_count=search_info.get("candidate_count", 0),
                )
            if i < 10:
                action_arr_before[i] = a_i

        next_state_img, _reward_total, terminated, truncated, info = env.step(
            {"hoprate": fixed_hoprate, "offsets": offsets}
        )

        ber_blocks = info.get("ber_blocks", [])
        per_block_rewards = compute_block_rewards(ber_blocks, info.get("hoprate_used", fixed_hoprate))
        ep_block_rewards.extend(per_block_rewards)

        mean_step_ber = np.mean(ber_blocks) if len(ber_blocks) > 0 else 0.0
        mean_step_reward = np.mean(per_block_rewards) if len(per_block_rewards) > 0 else 0.0

        arr_before = np.zeros(10, dtype=np.float32)
        for i in range(10):
            a_i = int(offsets[i])
            r_i = float(per_block_rewards[i])

            arr_after = arr_before.copy()
            if i < 10:
                arr_after[i] = a_i

            buffer.add(
                state_img,
                fixed_hoprate,
                arr_before,
                a_i,
                r_i,
                next_state_img,
                fixed_hoprate,
                arr_after,
                False,
            )

            if i < 10:
                arr_before[i] = a_i

        state_img = next_state_img

        train_stats = {}
        if buffer.size() >= args.min_buffer_before_train:
            for _ in range(args.update_iters_per_step):
                batch = buffer.sample(args.batch_size)
                eas_batch = None
                if args.teacher_from_replay and eas_buffer.size() >= args.eas_batch_size:
                    eas_batch = eas_buffer.sample(args.eas_batch_size)
                train_stats = agent.update(batch, eas_batch)

        search_stats = agent.consume_rollout_stats()
        step_duration = time.time() - step_start_time

        plot_rewards.append(mean_step_reward)
        plot_bers.append(mean_step_ber)
        plot_losses_actor.append(train_stats.get('actor_loss', 0) if train_stats else 0)
        plot_losses_critic.append(train_stats.get('critic1_loss', 0) if train_stats else 0)
        plot_losses_distill.append(train_stats.get('distill_loss', 0) if train_stats else 0)
        plot_search_change_rate.append(search_stats.get('search_change_rate', 0))
        plot_eas_valid_ratio.append(train_stats.get('eas_valid_ratio', 0) if train_stats else 0)

        if search_stats.get('search_total', 0) > 0:
            search_change_text = f"{search_stats.get('search_change_rate', 0):.2%}"
            search_gain_text = f"{search_stats.get('search_avg_gain', 0):.4f}"
        else:
            search_change_text = "N/A"
            search_gain_text = "N/A"

        log_msg = (f"Step {step_idx}/{args.steps_per_episode} | "
                   f"Offsets: {offsets.astype(int).tolist()} | "
                   f"Rew: {mean_step_reward:.4f} | "
                   f"BER: {mean_step_ber:.4f} | "
                   f"SearchChange: {search_change_text} | "
                   f"SearchGain: {search_gain_text} | "
                   f"EASBuf: {eas_buffer.size()}")

        if train_stats:
            log_msg += (f" | Loss: A={train_stats.get('actor_loss', 0):.3f}, "
                        f"C={train_stats.get('critic1_loss', 0):.3f}, "
                        f"D={train_stats.get('distill_loss', 0):.3f}, "
                        f"Alpha={train_stats.get('alpha', 0):.5f}, "
                        f"EASValid={train_stats.get('eas_valid_ratio', 0):.2%}, "
                        f"EASCount={train_stats.get('eas_valid_count', 0)}")

        log_msg += f" | T: {step_duration:.2f}s"
        logger.info(log_msg)

        if terminated or truncated:
            logger.info("Episode terminated early.")
            break

    mean_ep_reward = float(np.mean(ep_block_rewards)) if len(ep_block_rewards) > 0 else 0.0
    logger.info(f"--- Episode {episode} Finished ---")

    try:
        plt.figure()
        plt.plot(plot_rewards)
        plt.title("Mean Step Reward")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "reward.png"))
        plt.close()

        plt.figure()
        plt.plot(plot_bers, color='r')
        plt.title("Mean Step BER")
        plt.xlabel("Step")
        plt.ylabel("BER")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "ber.png"))
        plt.close()

        plt.figure()
        plt.plot(plot_losses_actor, label="Actor Loss", alpha=0.7)
        plt.plot(plot_losses_critic, label="Critic Loss", alpha=0.7)
        plt.plot(plot_losses_distill, label="Distill Loss", alpha=0.7)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "loss.png"))
        plt.close()

        plt.figure()
        plt.plot(plot_search_change_rate, label="Search Change Rate", color='g')
        plt.title("Local Search Change Rate")
        plt.xlabel("Step")
        plt.ylabel("Rate")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "search_change_rate.png"))
        plt.close()

        plt.figure()
        plt.plot(plot_eas_valid_ratio, label="EAS Valid Ratio", color='m')
        plt.title("EAS Distillation Valid Ratio")
        plt.xlabel("Step")
        plt.ylabel("Ratio")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "eas_valid_ratio.png"))
        plt.close()

        logger.info(f"Plots saved to {args.output_dir}.")
    except Exception as e:
        logger.error(f"Plotting failed: {e}")

    total_duration = time.time() - start_time
    logger.info(f"Total Time: {total_duration:.2f}s | Mean Ep Reward: {mean_ep_reward:.4f} | Episode Time: {time.time() - ep_start_time:.2f}s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_episode", type=int, default=settings.TRAIN_CONFIG["steps_per_episode"])
    parser.add_argument("--output_dir", type=str, default="outputs/eas_local")
    parser.add_argument("--log_file", type=str, default=settings.LOG_FILE)

    parser.add_argument("--actor_lr", type=float, default=settings.SAC_CONFIG["actor_lr"])
    parser.add_argument("--critic_lr", type=float, default=settings.SAC_CONFIG["critic_lr"])
    parser.add_argument("--alpha_lr", type=float, default=settings.SAC_CONFIG["alpha_lr"])
    parser.add_argument("--tau", type=float, default=settings.SAC_CONFIG["tau"])
    parser.add_argument("--gamma", type=float, default=settings.SAC_CONFIG["gamma"])

    parser.add_argument("--search_radius", type=int, default=settings.EAS_LOCAL_CONFIG["search_radius"])
    parser.add_argument("--distill_coef", type=float, default=settings.EAS_LOCAL_CONFIG["distill_coef"])
    parser.add_argument("--search_eval", type=str, default=settings.EAS_LOCAL_CONFIG["search_eval"])
    parser.add_argument("--teacher_from_replay", action="store_true", default=settings.EAS_LOCAL_CONFIG["teacher_from_replay"])
    parser.add_argument("--no_teacher_from_replay", dest="teacher_from_replay", action="store_false")
    parser.add_argument("--log_search_stats", action="store_true", default=settings.EAS_LOCAL_CONFIG["log_search_stats"])
    parser.add_argument("--no_log_search_stats", dest="log_search_stats", action="store_false")
    parser.add_argument("--filter_teacher_on_update", action="store_true", default=settings.EAS_LOCAL_CONFIG["filter_teacher_on_update"])
    parser.add_argument("--no_filter_teacher_on_update", dest="filter_teacher_on_update", action="store_false")
    parser.add_argument("--teacher_compare_mode", type=str, default=settings.EAS_LOCAL_CONFIG["teacher_compare_mode"])

    parser.add_argument("--replay_size", type=int, default=settings.BUFFER_CONFIG["capacity"])
    parser.add_argument("--batch_size", type=int, default=settings.BUFFER_CONFIG["batch_size"])
    parser.add_argument("--eas_replay_capacity", type=int, default=settings.EAS_LOCAL_CONFIG["eas_replay_capacity"])
    parser.add_argument("--eas_batch_size", type=int, default=settings.EAS_LOCAL_CONFIG["eas_batch_size"])
    parser.add_argument("--min_buffer_before_train", type=int, default=settings.TRAIN_CONFIG["min_buffer_before_train"])
    parser.add_argument("--update_iters_per_step", type=int, default=settings.TRAIN_CONFIG["update_iters_per_step"])

    parser.add_argument("--cpu_only", action="store_true", default=settings.CPU_ONLY)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
