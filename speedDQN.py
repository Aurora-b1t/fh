import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from newray import FHSSQPSKEnv  # 假定与本文件在同一路径下


# -----------------------------
# 配置
# -----------------------------
@dataclass
class DQNConfig:
    state_dim: int = 10
    action_dim: int = 20  # 对应跳速 {20, 40， ..., 400}
    gamma: float = 0.99
    lr: float = 0.005
    batch_size: int = 64
    replay_size: int = 5000
    min_replay_size: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1_000
    target_update_interval: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# 网络
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 经验回放
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# -----------------------------
# DQN 智能体
# -----------------------------
class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.q_net = QNetwork(cfg.state_dim, cfg.action_dim).to(self.device)
        self.target_net = QNetwork(cfg.state_dim, cfg.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)

        self.replay = ReplayBuffer(cfg.replay_size)
        self.steps_done = 0

        # 预生成固定状态（20 维，恒定 0.1）
        self.fixed_state = np.full(cfg.state_dim, 0.1, dtype=np.float32)

        # 动作映射：索引 -> hoprate
        self.action_values = np.arange(cfg.action_dim+1, dtype=np.float32) * 20.0

    def select_action(self, state: np.ndarray):
        eps = self._epsilon_by_step(self.steps_done)
        self.steps_done += 1

        if random.random() < eps:
            action_idx = random.randrange(self.cfg.action_dim)
        else:
            with torch.no_grad():
                s = torch.as_tensor(state, device=self.device).unsqueeze(0)
                q_values = self.q_net(s)
                action_idx = int(q_values.argmax(dim=1).item())
        return action_idx

    def _epsilon_by_step(self, step: int):
        eps = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
              max(0, (self.cfg.epsilon_decay_steps - step) / self.cfg.epsilon_decay_steps)
        return eps

    def train_step(self):
        if len(self.replay) < self.cfg.min_replay_size:
            return 0.0

        state, action, reward, next_state, done = self.replay.sample(self.cfg.batch_size)

        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        next_state = torch.as_tensor(next_state, device=self.device, dtype=torch.float32)
        action = torch.as_tensor(action, device=self.device, dtype=torch.long)
        reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        done = torch.as_tensor(done, device=self.device, dtype=torch.float32)

        q_pred = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_state).max(dim=1).values
            q_target = reward + (1.0 - done) * self.cfg.gamma * next_q

        loss = nn.functional.mse_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())