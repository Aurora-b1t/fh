"""
Discrete-action Soft Actor-Critic (SAC) with convolutional state encoder.

The policy and critics use a PSD image plus two scalar features:
``hoprate`` and ``block_idx``.  ``block_idx`` records which of the 10 blocks in
an environment step the current offset decision belongs to.
"""

import collections
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import UninitializedParameter


EXTRA_DIM = 2


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

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
        self.buffer.append((
            np.array(state_img, dtype=np.float32),
            float(hoprate),
            float(block_idx),
            int(action),
            float(reward),
            np.array(next_state_img, dtype=np.float32),
            float(next_hoprate),
            float(next_block_idx),
            bool(done),
        ))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        (
            state_imgs,
            hoprates,
            block_idxs,
            actions,
            rewards,
            next_state_imgs,
            next_hoprates,
            next_block_idxs,
            dones,
        ) = zip(*transitions)
        return {
            "state_imgs": np.array(state_imgs, dtype=np.float32),
            "hoprates": np.array(hoprates, dtype=np.float32),
            "block_idxs": np.array(block_idxs, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_state_imgs": np.array(next_state_imgs, dtype=np.float32),
            "next_hoprates": np.array(next_hoprates, dtype=np.float32),
            "next_block_idxs": np.array(next_block_idxs, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
        }

    def size(self):
        return len(self.buffer)


class EASReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(
        self,
        state_img,
        hoprate,
        block_idx,
        teacher_action,
        search_gain=0.0,
        candidate_count=0,
    ):
        self.buffer.append((
            np.array(state_img, dtype=np.float32),
            float(hoprate),
            float(block_idx),
            int(teacher_action),
            float(search_gain),
            int(candidate_count),
        ))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        (
            state_imgs,
            hoprates,
            block_idxs,
            teacher_actions,
            search_gains,
            candidate_counts,
        ) = zip(*transitions)
        return {
            "state_imgs": np.array(state_imgs, dtype=np.float32),
            "hoprates": np.array(hoprates, dtype=np.float32),
            "block_idxs": np.array(block_idxs, dtype=np.float32),
            "teacher_actions": np.array(teacher_actions, dtype=np.int64),
            "search_gains": np.array(search_gains, dtype=np.float32),
            "candidate_counts": np.array(candidate_counts, dtype=np.int64),
        }

    def size(self):
        return len(self.buffer)


class PolicyNet(nn.Module):
    def __init__(self, n_actions, extra_embedding_module=None):
        super().__init__()
        self.conv1 = nn.LazyConv2d(out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.conv_fc = nn.LazyLinear(256)

        if extra_embedding_module is not None:
            self.extra_embedding = extra_embedding_module
        else:
            self.extra_embedding = nn.Sequential(
                nn.Linear(EXTRA_DIM, 64),
                nn.ReLU(),
            )

        self.fc_hidden = nn.LazyLinear(256)
        self.fc_logits = nn.Linear(256, n_actions)

        self.apply(self._init_weights)
        nn.init.uniform_(self.fc_logits.weight, -0.003, 0.003)
        nn.init.zeros_(self.fc_logits.bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.LazyLinear, nn.LazyConv2d)):
            if isinstance(getattr(m, "weight", None), UninitializedParameter):
                return
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight, 0, 0.1)
            if getattr(m, "bias", None) is not None and not isinstance(m.bias, UninitializedParameter):
                nn.init.zeros_(m.bias)

    def forward(self, img, extra):
        x = F.relu(self.bn1(self.conv1(img)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.conv_fc(x))
        extra = self.extra_embedding(extra)
        x = torch.cat([x, extra], dim=1)
        x = F.relu(self.fc_hidden(x))
        logits = self.fc_logits(x)
        probs = F.softmax(logits, dim=1)
        return probs, logits


class ValueNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.LazyConv2d(out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.conv_fc = nn.LazyLinear(256)

        self.extra_embedding = nn.Sequential(
            nn.Linear(EXTRA_DIM, 64),
            nn.ReLU(),
        )

        self.fc_hidden = nn.LazyLinear(256)
        self.fc_out = nn.Linear(256, n_actions)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.LazyLinear, nn.LazyConv2d)):
            if isinstance(getattr(m, "weight", None), UninitializedParameter):
                return
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight, 0, 0.1)
            if getattr(m, "bias", None) is not None and not isinstance(m.bias, UninitializedParameter):
                nn.init.zeros_(m.bias)

    def forward(self, img, extra):
        extra = self.extra_embedding(extra)
        x = F.relu(self.bn1(self.conv1(img)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.conv_fc(x))
        x = torch.cat([x, extra], dim=1)
        return self.fc_out(F.relu(self.fc_hidden(x)))


class SAC:
    def __init__(
        self,
        n_actions,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
    ):
        self.critic_1 = ValueNet(n_actions).to(device)
        self.critic_2 = ValueNet(n_actions).to(device)
        self.actor = PolicyNet(
            n_actions,
            extra_embedding_module=self.critic_1.extra_embedding,
        ).to(device)

        self.target_critic_1 = ValueNet(n_actions).to(device)
        self.target_critic_2 = ValueNet(n_actions).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        actor_params = [
            p for name, p in self.actor.named_parameters()
            if "extra_embedding" not in name
        ]
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(
            np.log(0.01),
            dtype=torch.float,
            device=device,
            requires_grad=True,
        )
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.n_actions = n_actions

    def take_action(self, state_img_np, hoprate, block_idx):
        img = torch.tensor(
            state_img_np,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).unsqueeze(0)
        extra_np = np.array([hoprate, block_idx], dtype=np.float32)
        extra = torch.tensor(extra_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            probs, _ = self.actor(img, extra)
            action = torch.distributions.Categorical(probs).sample().item()
        if was_training:
            self.actor.train()
        return action

    def take_action_sequence(self, state_img_np, hoprate):
        actions = []
        for block_idx in range(10):
            actions.append(self.take_action(state_img_np, hoprate, block_idx))
        return actions

    def _build_extra_tensor(self, hoprates, block_idxs):
        hoprates = torch.tensor(hoprates, dtype=torch.float32, device=self.device).view(-1, 1)
        block_idxs = torch.tensor(block_idxs, dtype=torch.float32, device=self.device).view(-1, 1)
        return torch.cat([hoprates, block_idxs], dim=1)

    def calc_target(self, rewards, next_imgs, next_extras, dones):
        next_probs, _ = self.actor(next_imgs, next_extras)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)

        q1 = self.target_critic_1(next_imgs, next_extras)
        q2 = self.target_critic_2(next_imgs, next_extras)
        min_q = torch.sum(next_probs * torch.min(q1, q2), dim=1, keepdim=True)
        next_value = min_q + self.log_alpha.exp() * entropy
        return rewards + self.gamma * next_value * (1 - dones)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        imgs = torch.tensor(
            transition_dict["state_imgs"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_imgs = torch.tensor(
            transition_dict["next_state_imgs"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        extras = self._build_extra_tensor(
            transition_dict["hoprates"],
            transition_dict["block_idxs"],
        )
        next_extras = self._build_extra_tensor(
            transition_dict["next_hoprates"],
            transition_dict["next_block_idxs"],
        )

        actions = torch.tensor(
            transition_dict["actions"],
            dtype=torch.long,
            device=self.device,
        ).view(-1, 1)
        rewards = torch.tensor(
            transition_dict["rewards"],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        dones = torch.tensor(
            transition_dict["dones"],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)

        td_target = self.calc_target(rewards, next_imgs, next_extras, dones)
        q1_pred = self.critic_1(imgs, extras).gather(1, actions)
        q2_pred = self.critic_2(imgs, extras).gather(1, actions)
        critic_1_loss = F.mse_loss(q1_pred, td_target.detach())
        critic_2_loss = F.mse_loss(q2_pred, td_target.detach())

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        probs, _ = self.actor(imgs, extras)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)

        q1 = self.critic_1(imgs, extras)
        q2 = self.critic_2(imgs, extras)
        min_q = torch.sum(probs * torch.min(q1, q2), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_q)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean(self.log_alpha * (entropy.detach() - self.target_entropy))
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        return {
            "critic1_loss": critic_1_loss.item(),
            "critic2_loss": critic_2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": float(self.log_alpha.exp().item()),
        }
