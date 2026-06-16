"""
EAS local-neighborhood-search variant of discrete SAC.

Extends baseline SAC with critic-guided local action search during rollout and
teacher-action distillation during updates via a separate EAS replay buffer.
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from SAC import PolicyNet, ValueNet


class SACEASLocal:
    def __init__(self, n_actions,
                 actor_lr, critic_lr, alpha_lr,
                 target_entropy, tau, gamma, device,
                 search_radius=1,
                 distill_coef=0.1,
                 search_eval="min_q",
                 teacher_from_replay=True,
                 log_search_stats=True,
                 filter_teacher_on_update=True,
                 teacher_compare_mode="min_q"):
        self.critic_1 = ValueNet(n_actions).to(device)
        self.critic_2 = ValueNet(n_actions).to(device)
        self.actor = PolicyNet(n_actions, extra_embedding_module=self.critic_1.extra_embedding).to(device)

        self.target_critic_1 = ValueNet(n_actions).to(device)
        self.target_critic_2 = ValueNet(n_actions).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

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

        self.search_radius = int(search_radius)
        self.distill_coef = float(distill_coef)
        self.search_eval = search_eval
        self.teacher_from_replay = bool(teacher_from_replay)
        self.log_search_stats = bool(log_search_stats)
        self.filter_teacher_on_update = bool(filter_teacher_on_update)
        self.teacher_compare_mode = teacher_compare_mode

        self._reset_rollout_stats()

    def _reset_rollout_stats(self):
        self._search_total = 0
        self._search_changed = 0
        self._search_gain_sum = 0.0

    def consume_rollout_stats(self):
        total = self._search_total
        changed = self._search_changed
        gain = self._search_gain_sum
        self._reset_rollout_stats()
        if total <= 0:
            return {
                "search_change_rate": 0.0,
                "search_avg_gain": 0.0,
                "search_total": 0,
            }
        return {
            "search_change_rate": changed / total,
            "search_avg_gain": gain / total,
            "search_total": total,
        }

    def _build_extra_tensor(self, hoprates, action_arrs):
        hoprates = torch.tensor(hoprates, dtype=torch.float32, device=self.device).view(-1, 1)
        action_arrs = torch.tensor(action_arrs, dtype=torch.float32, device=self.device)
        return torch.cat([hoprates, action_arrs], dim=1)

    def _state_to_tensors(self, state_img_np, hoprate, action_arr_np):
        img = torch.tensor(state_img_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        extra_np = np.concatenate(([hoprate], np.array(action_arr_np, dtype=np.float32))).astype(np.float32)
        extra = torch.tensor(extra_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return img, extra

    def build_local_candidates(self, seed_action):
        seed_action = int(seed_action)
        if self.search_radius <= 0 or self.n_actions <= 1:
            return []

        candidates = []
        seen = set()
        for delta in range(1, self.search_radius + 1):
            for candidate in (
                (seed_action - delta) % self.n_actions,
                (seed_action + delta) % self.n_actions,
            ):
                if candidate == seed_action or candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(int(candidate))
        return candidates

    def _score_candidates(self, q1_values, q2_values, candidates, eval_mode=None):
        eval_mode = eval_mode or self.search_eval
        if eval_mode != "min_q":
            raise ValueError(f"Unsupported search_eval: {eval_mode}")
        q_min = torch.min(q1_values, q2_values)
        candidate_scores = q_min[candidates]
        best_local_idx = torch.argmax(candidate_scores).item()
        best_action = int(candidates[best_local_idx])
        best_score = float(candidate_scores[best_local_idx].item())
        return best_action, best_score, q_min

    def local_search_action(self, state_img_np, hoprate, action_arr_np):
        img, extra = self._state_to_tensors(state_img_np, hoprate, action_arr_np)

        actor_was_training = self.actor.training
        critic1_was_training = self.critic_1.training
        critic2_was_training = self.critic_2.training
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

        with torch.no_grad():
            probs, _ = self.actor(img, extra)
            probs = probs.squeeze(0)
            seed_action = torch.distributions.Categorical(probs).sample().item()

            q1_values = self.critic_1(img, extra).squeeze(0)
            q2_values = self.critic_2(img, extra).squeeze(0)
            q_min = torch.min(q1_values, q2_values)
            seed_score = float(q_min[seed_action].item())
            candidates = self.build_local_candidates(seed_action)
            if len(candidates) > 0:
                best_candidate_action, best_candidate_score, _ = self._score_candidates(q1_values, q2_values, candidates)
                if best_candidate_score > seed_score:
                    best_action = int(best_candidate_action)
                    best_score = float(best_candidate_score)
                else:
                    best_action = int(seed_action)
                    best_score = seed_score
            else:
                best_action = int(seed_action)
                best_score = seed_score

        if actor_was_training:
            self.actor.train()
        if critic1_was_training:
            self.critic_1.train()
        if critic2_was_training:
            self.critic_2.train()

        self._search_total += 1
        if best_action != seed_action:
            self._search_changed += 1
        self._search_gain_sum += (best_score - seed_score)

        return seed_action, {
            "seed_action": int(seed_action),
            "teacher_action": int(best_action),
            "selected_action": int(best_action),
            "search_gain": float(best_score - seed_score),
            "candidate_count": len(candidates),
        }

    def take_action(self, state_img_np, hoprate, action_arr_np):
        img, extra = self._state_to_tensors(state_img_np, hoprate, action_arr_np)

        actor_was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            probs, _ = self.actor(img, extra)
            probs = probs.squeeze(0)
            action = torch.distributions.Categorical(probs).sample().item()
        if actor_was_training:
            self.actor.train()
        return int(action)

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

    def _calc_eas_distill_loss(self, eas_transition_dict):
        zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        if not self.teacher_from_replay or eas_transition_dict is None:
            return zero, 0.0, 0

        imgs = torch.tensor(eas_transition_dict['state_imgs'], dtype=torch.float32, device=self.device).unsqueeze(1)
        extras = self._build_extra_tensor(eas_transition_dict['hoprates'], eas_transition_dict['action_arrs'])
        teacher_actions = torch.tensor(eas_transition_dict['teacher_actions'], dtype=torch.long, device=self.device)

        probs, logits = self.actor(imgs, extras)
        ce_losses = F.cross_entropy(logits, teacher_actions, reduction='none')

        q1_values = self.critic_1(imgs, extras)
        q2_values = self.critic_2(imgs, extras)
        _, current_actor_actions = torch.max(probs, dim=1)

        compare_mode = self.teacher_compare_mode or self.search_eval
        if compare_mode != "min_q":
            raise ValueError(f"Unsupported teacher_compare_mode: {compare_mode}")
        q_min = torch.min(q1_values, q2_values)
        teacher_scores = q_min.gather(1, teacher_actions.view(-1, 1)).squeeze(1)
        actor_scores = q_min.gather(1, current_actor_actions.view(-1, 1)).squeeze(1)

        if self.filter_teacher_on_update:
            valid_mask = teacher_scores > actor_scores
        else:
            valid_mask = torch.ones_like(teacher_scores, dtype=torch.bool)

        valid_count = int(valid_mask.sum().item())
        if valid_count <= 0:
            return zero, 0.0, 0

        masked_loss = ce_losses[valid_mask].mean()
        valid_ratio = float(valid_count / teacher_actions.shape[0])
        return masked_loss, valid_ratio, valid_count

    def update(self, transition_dict, eas_transition_dict=None):
        imgs = torch.tensor(transition_dict['state_imgs'], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_imgs = torch.tensor(transition_dict['next_state_imgs'], dtype=torch.float32, device=self.device).unsqueeze(1)
        extras = self._build_extra_tensor(transition_dict['hoprates'], transition_dict['action_arrs'])
        next_extras = self._build_extra_tensor(transition_dict['next_hoprates'], transition_dict['next_action_arrs'])

        actions = torch.tensor(transition_dict['actions'], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32, device=self.device).view(-1, 1)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32, device=self.device).view(-1, 1)

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
        sac_actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_q)

        distill_loss, eas_valid_ratio, eas_valid_count = self._calc_eas_distill_loss(eas_transition_dict)

        actor_loss = sac_actor_loss + self.distill_coef * distill_loss
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
            "sac_actor_loss": sac_actor_loss.item(),
            "distill_loss": distill_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": float(self.log_alpha.exp().item()),
            "eas_valid_ratio": eas_valid_ratio,
            "eas_valid_count": eas_valid_count,
        }
