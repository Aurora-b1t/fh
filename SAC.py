import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import UninitializedParameter
import numpy as np
import collections
import random

# ----------------------------------------- #
# 经验回放池（扩展：存储 hoprate 和 10 维 action_arr）
# 每条经验对应一次离散动作决策：
# (state_img, hoprate, action_arr_before, action, reward, next_state_img, next_hoprate, next_action_arr, done)
# ----------------------------------------- #
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state_img, hoprate, action_arr, action, reward,
            next_state_img, next_hoprate, next_action_arr, done):
        # action_arr 是当前决策前的 10 维向量（上一些动作的记录）
        self.buffer.append((
            np.array(state_img, dtype=np.float32),
            float(hoprate),
            np.array(action_arr, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state_img, dtype=np.float32),
            float(next_hoprate),
            np.array(next_action_arr, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        (state_imgs, hoprates, action_arrs, actions, rewards,
         next_state_imgs, next_hoprates, next_action_arrs, dones) = zip(*transitions)
        return {
            "state_imgs": np.array(state_imgs, dtype=np.float32),               # [B, H, W]
            "hoprates": np.array(hoprates, dtype=np.float32),                   # [B]
            "action_arrs": np.array(action_arrs, dtype=np.float32),             # [B, 10]
            "actions": np.array(actions, dtype=np.int64),                       # [B]
            "rewards": np.array(rewards, dtype=np.float32),                     # [B]
            "next_state_imgs": np.array(next_state_imgs, dtype=np.float32),     # [B, H, W]
            "next_hoprates": np.array(next_hoprates, dtype=np.float32),         # [B]
            "next_action_arrs": np.array(next_action_arrs, dtype=np.float32),   # [B, 10]
            "dones": np.array(dones, dtype=np.float32),                         # [B]
        }

    def size(self):
        return len(self.buffer)


# ----------------------------------------- #
# 卷积 + 额外特征版策略网络
# 额外特征 extra = [hoprate(1), action_arr(10)] 共 11 维，不经过卷积
# 与卷积特征拼接后进入后续全连接层
# 输出对 n_actions 的概率分布
# ----------------------------------------- #
class PolicyNet(nn.Module):
    def __init__(self, n_actions, extra_embedding_module=None):
        super().__init__()
        # 卷积支路
        self.conv1 = nn.LazyConv2d(out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.conv_fc = nn.LazyLinear(256)  # 将卷积特征压到 256

        # 额外特征embedding
        if extra_embedding_module is not None:
            self.extra_embedding = extra_embedding_module
        else:
            self.extra_embedding = nn.Sequential(
                nn.Linear(11, 64),
                nn.ReLU()
            )

        # 融合后续 MLP（输入为 [conv_fc(256) + extra(11)]，使用 LazyLinear 自适应首轮输入维度）
        self.fc_hidden = nn.LazyLinear(256)
        self.fc_logits = nn.Linear(256, n_actions)

        self.apply(self._init_weights)

        # ---------------------------------------------------- #
        # 重要修正：将策略网络最后一层初始化为接近 0
        # 这样初始输出的 Logits 接近 0，Softmax 后概率接近均匀分布
        # 避免一开始就“死板”地只选某一个动作
        # ---------------------------------------------------- #
        nn.init.uniform_(self.fc_logits.weight, -0.003, 0.003)
        nn.init.zeros_(self.fc_logits.bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.LazyLinear, nn.LazyConv2d)):
            # Lazy 模块在首个 forward 前未初始化，跳过
            if isinstance(getattr(m, 'weight', None), UninitializedParameter):
                return
            if getattr(m, 'weight', None) is not None:
                nn.init.normal_(m.weight, 0, 0.1)
            if getattr(m, 'bias', None) is not None and not isinstance(m.bias, UninitializedParameter):
                nn.init.zeros_(m.bias)

    def forward(self, img, extra):  # img: [B, 1, H, W], extra: [B, 11]
        x = F.relu(self.bn1(self.conv1(img)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.conv_fc(x))
        # 拼接额外特征
        extra = self.extra_embedding(extra)
        x = torch.cat([x, extra], dim=1)
        x = F.relu(self.fc_hidden(x))
        logits = self.fc_logits(x)
        probs = F.softmax(logits, dim=1)
        return probs, logits


# ----------------------------------------- #
# 卷积 + 额外特征版价值网络（Q 网络）
# 输入与策略网络一致，输出对每个离散动作的 Q 值
# ----------------------------------------- #
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
            nn.Linear(11, 64),
            nn.ReLU()
        )

        self.fc_hidden = nn.LazyLinear(256)  # 输入 [256 + 11]
        self.fc_out = nn.Linear(256, n_actions)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.LazyLinear, nn.LazyConv2d)):
            if isinstance(getattr(m, 'weight', None), UninitializedParameter):
                return
            if getattr(m, 'weight', None) is not None:
                nn.init.normal_(m.weight, 0, 0.1)
            if getattr(m, 'bias', None) is not None and not isinstance(m.bias, UninitializedParameter):
                nn.init.zeros_(m.bias)

    def forward(self, img, extra):  # img: [B, 1, H, W], extra: [B, 11]
        extra = self.extra_embedding(extra)
        x = F.relu(self.bn1(self.conv1(img)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.conv_fc(x))
        x = torch.cat([x, extra], dim=1)
        x = F.relu(self.fc_hidden(x))
        q = self.fc_out(x)
        return q


# ----------------------------------------- #
# SAC 主体（离散动作，扩展状态：图像 + hoprate + 10维 action_arr）
# ----------------------------------------- #
class SAC:
    def __init__(self, n_actions,
                 actor_lr, critic_lr, alpha_lr,
                 target_entropy, tau, gamma, device):
        self.critic_1 = ValueNet(n_actions).to(device)
        self.critic_2 = ValueNet(n_actions).to(device)
        # 共享 Critic 1 的 extra_embedding
        self.actor = PolicyNet(n_actions, extra_embedding_module=self.critic_1.extra_embedding).to(device)
        
        self.target_critic_1 = ValueNet(n_actions).to(device)
        self.target_critic_2 = ValueNet(n_actions).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Actor 优化器：过滤掉共享的 extra_embedding 参数，防止 Actor 为了优化策略破坏 Critic 学到的特征
        actor_params = [p for name, p in self.actor.named_parameters() if "extra_embedding" not in name]
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=actor_lr)
        
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, device=device, requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.n_actions = n_actions

    # 单次决策：给定图像、hoprate 和 当前 10维 action_arr，输出一个动作
    def take_action(self, state_img_np, hoprate, action_arr_np):
        img = torch.tensor(state_img_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        extra_np = np.concatenate(([hoprate], np.array(action_arr_np, dtype=np.float32))).astype(np.float32)  # [11]
        extra = torch.tensor(extra_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,11]

        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            probs, _ = self.actor(img, extra)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
        if was_training:
            self.actor.train()
        return action

    # 一次“步”（一个 100x100 矩阵 + hoprate），顺序输出 10 个动作
    # 每次输出后，按顺序写入 action_arr 的第 i 个位置（i=0..9）
    def take_action_sequence(self, state_img_np, hoprate):
        #hoprate压缩成与action_arr相似的范围
        hoprate = hoprate / 50
        actions = []
        action_arr = np.zeros(10, dtype=np.float32)
        for i in range(10):
            a = self.take_action(state_img_np, hoprate, action_arr)
            actions.append(a)
            if i < 10:
                action_arr[i] = a
        return actions  # 长度 10 的动作序列

    def _build_extra_tensor(self, hoprates, action_arrs):
        # hoprates: [B], action_arrs: [B,10] -> extra: [B,11]
        hoprates = torch.tensor(hoprates, dtype=torch.float32, device=self.device).view(-1, 1)
        action_arrs = torch.tensor(action_arrs, dtype=torch.float32, device=self.device)
        extra = torch.cat([hoprates, action_arrs], dim=1)
        return extra

    def calc_target(self, rewards, next_imgs, next_extras, dones):
        next_probs, _ = self.actor(next_imgs, next_extras)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)

        q1 = self.target_critic_1(next_imgs, next_extras)
        q2 = self.target_critic_2(next_imgs, next_extras)
        min_q = torch.sum(next_probs * torch.min(q1, q2), dim=1, keepdim=True)
        next_value = min_q + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        # 期望输入键：
        # state_imgs, hoprates, action_arrs, actions, rewards, next_state_imgs, next_hoprates, next_action_arrs, dones
        imgs = torch.tensor(transition_dict['state_imgs'], dtype=torch.float32, device=self.device).unsqueeze(1)          # [B,1,H,W]
        next_imgs = torch.tensor(transition_dict['next_state_imgs'], dtype=torch.float32, device=self.device).unsqueeze(1)# [B,1,H,W]
        extras = self._build_extra_tensor(transition_dict['hoprates'], transition_dict['action_arrs'])                    # [B,11]
        next_extras = self._build_extra_tensor(transition_dict['next_hoprates'], transition_dict['next_action_arrs'])     # [B,11]

        actions = torch.tensor(transition_dict['actions'], dtype=torch.long, device=self.device).view(-1, 1)              # [B,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32, device=self.device).view(-1, 1)           # [B,1]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32, device=self.device).view(-1, 1)               # [B,1]

        # Critic 更新
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

        # Actor 更新
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

        # 温度系数 alpha 更新
        # 离散 SAC 中 target_entropy 为正熵目标。最小化该 loss 时：
        # entropy < target_entropy -> log_alpha 增大，鼓励探索；反之减小。
        alpha_loss = torch.mean(self.log_alpha * (entropy.detach() - self.target_entropy))
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        return {
            "critic1_loss": critic_1_loss.item(),
            "critic2_loss": critic_2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": float(self.log_alpha.exp().item()),
        }