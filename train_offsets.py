import argparse
import os
import time
import numpy as np
import torch

from SAC import SAC, ReplayBuffer
from fh_env import FHSSQPSKEnv


def build_agent_and_env(args):
    # 创建环境（可根据需要调整干扰/衰落等开关）
    env = FHSSQPSKEnv(
        enable_reactive=True,
        enable_sweep=True,
        enable_rayleigh=True,
        debug_plot_psd=False,
        debug_log_hops=False,
    )

    # 离散动作数量与子信道数量一致（offset 取值范围 0..num_channels-1）
    n_actions = env.num_channels

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")

    # 构建 SAC
    # target_entropy 设为 0.9 * log(n_actions)（鼓励较高但非最大熵）
    agent = SAC(
        n_actions=n_actions,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        target_entropy=np.log(n_actions) * 0.9,
        tau=args.tau,
        gamma=args.gamma,
        device=device,
    )

    # 经验回放
    buffer = ReplayBuffer(capacity=args.replay_size)

    return env, agent, buffer, device, n_actions


def compute_block_rewards(ber_blocks, hoprate):
    # 与环境总奖励一致的分块版本：reward_i = 0.5 - ber_i - 1e-4 * hoprate
    rewards = [0.5 - float(ber) - 1e-4 * float(hoprate) for ber in ber_blocks]
    return rewards


def train(args):
    env, agent, buffer, device, n_actions = build_agent_and_env(args)

    # 固定 hoprate
    fixed_hoprate = 200.0

    # 训练循环
    total_steps = 0
    for episode in range(1, args.episodes + 1):
        # 环境 reset 返回 100ms 观测瀑布图
        state_img, info = env.reset()
        done = False

        ep_block_rewards = []

        # 每个 episode 内执行若干 env.step（每次 step 含 10 个动作）
        for step_idx in range(args.steps_per_episode):
            # 逐块依次决策 10 个 offset
            offsets = np.zeros(10, dtype=np.float32)
            action_arr_before = np.zeros(10, dtype=np.float32)  # 存储前 10 个动作的记录

            for i in range(10):
                a_i = agent.take_action(state_img, fixed_hoprate, action_arr_before)
                a_i = int(np.clip(a_i, 0, n_actions - 1))  # 保证在 0..num_channels-1
                offsets[i] = a_i
                if i < 10:
                    action_arr_before[i] = a_i

            # 与环境交互（整段 1s 传输）
            next_state_img, _reward_total, terminated, truncated, info = env.step(
                {"hoprate": fixed_hoprate, "offsets": offsets}
            )

            # 从 info 中获取每块 BER，计算每块 reward
            ber_blocks = info.get("ber_blocks", [])
            per_block_rewards = compute_block_rewards(ber_blocks, info.get("hoprate_used", fixed_hoprate))
            ep_block_rewards.extend(per_block_rewards)

            # 将 10 个动作分别作为 10 条经验写入回放池
            # 每条经验的 next_state_img 相同（为 step 后的 100ms 观测），符合当前环境设计
            arr_before = np.zeros(10, dtype=np.float32)  # 本条经验的前态动作记录
            for i in range(10):
                a_i = int(offsets[i])
                r_i = float(per_block_rewards[i])

                arr_after = arr_before.copy()
                if i < 10:
                    arr_after[i] = a_i

                buffer.add(
                    state_img,                 # 当前图像状态（100ms 瀑布图）
                    fixed_hoprate,             # 当前 hoprate（恒定 200）
                    arr_before,                # 本条经验决策前的 10 维动作记录
                    a_i,                       # 当前离散动作（offset）
                    r_i,                       # 针对本 block 的 reward
                    next_state_img,            # 下一图像状态（step 后 100ms 观测）
                    fixed_hoprate,             # 下一 hoprate（同为 200）
                    arr_after,                 # 下一动作记录（写入当前动作）
                    False,                     # 当前环境为持续交互，不终止
                )

                # 更新下一条经验的“前态动作记录”
                if i < 10:
                    arr_before[i] = a_i

            # 状态推进
            state_img = next_state_img
            total_steps += 1

            # 更新网络
            if buffer.size() >= args.min_buffer_before_train:
                for _ in range(args.update_iters_per_step):
                    batch = buffer.sample(args.batch_size)
                    stats = agent.update(batch)

            if terminated or truncated:
                break

        mean_ep_reward = float(np.mean(ep_block_rewards)) if len(ep_block_rewards) > 0 else 0.0
        print(f"Episode {episode:04d} | steps: {total_steps:05d} | mean_block_reward: {mean_ep_reward:.4f} | mean_BER: {info.get('mean_ber', np.nan):.4f}")

    print("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100, help="训练的 episode 数量")
    parser.add_argument("--steps_per_episode", type=int, default=10, help="每个 episode 内执行的 env.step 次数（每次 step 含 10 个动作）")

    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--min_buffer_before_train", type=int, default=1000, help="开始训练前的最小缓冲条目数")
    parser.add_argument("--update_iters_per_step", type=int, default=1, help="每次 env.step 后的更新次数")

    parser.add_argument("--cpu_only", action="store_true", help="仅使用 CPU 运行")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)