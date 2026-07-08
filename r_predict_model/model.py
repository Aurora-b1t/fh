"""
Reward-only ensemble model used by the MBPO training entry point.

The original MBPO template predicted reward plus state deltas for continuous
control.  In this project the SAC state is a PSD image plus the current block
index, and synthetic rollouts are intentionally limited to one step.
The model therefore learns only:

    flattened(state_img, hoprate, block_idx, action) -> reward
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardScaler:
    def __init__(self):
        self.mu = None
        self.std = None

    def fit(self, data):
        data = np.asarray(data, dtype=np.float32)
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        if self.mu is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transform().")
        return (np.asarray(data, dtype=np.float32) - self.mu) / self.std

    def inverse_transform(self, data):
        if self.mu is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform().")
        return self.std * data + self.mu


def _truncated_normal_(tensor, mean=0.0, std=0.01):
    with torch.no_grad():
        torch.nn.init.normal_(tensor, mean=mean, std=std)
        while True:
            cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
            if not torch.sum(cond):
                break
            tensor[cond] = torch.normal(
                mean=mean,
                std=std,
                size=(int(cond.sum().item()),),
                device=tensor.device,
            )


def init_weights(module):
    if isinstance(module, (nn.Linear, EnsembleFC)):
        input_dim = module.in_features
        _truncated_normal_(module.weight, std=1 / (2 * np.sqrt(input_dim)))
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        ensemble_size=5,
        weight_decay=0.0,
        bias=True,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.ensemble_size = int(ensemble_size)
        self.weight_decay = float(weight_decay)
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        _truncated_normal_(self.weight, std=1 / (2 * np.sqrt(self.in_features)))
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, input_tensor):
        output = torch.bmm(input_tensor, self.weight)
        if self.bias is not None:
            output = output + self.bias[:, None, :]
        return output


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class EnsembleModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        ensemble_size,
        hidden_size=200,
        learning_rate=1e-3,
        use_decay=False,
        device=None,
    ):
        super().__init__()
        self.output_dim = int(output_size)
        self.use_decay = bool(use_decay)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.nn1 = EnsembleFC(input_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.max_logvar = nn.Parameter(
            torch.ones((1, self.output_dim), device=self.device) / 2,
            requires_grad=False,
        )
        self.min_logvar = nn.Parameter(
            -torch.ones((1, self.output_dim), device=self.device) * 10,
            requires_grad=False,
        )
        self.swish = Swish()
        self.apply(init_weights)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, ret_log_var=False):
        x = self.swish(self.nn1(x))
        x = self.swish(self.nn2(x))
        x = self.swish(self.nn3(x))
        x = self.swish(self.nn4(x))
        output = self.nn5(x)
        mean = output[:, :, :self.output_dim]
        logvar = self.max_logvar - F.softplus(self.max_logvar - output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        if ret_log_var:
            return mean, logvar
        return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.0
        for module in self.children():
            if isinstance(module, EnsembleFC):
                decay_loss += module.weight_decay * torch.sum(torch.square(module.weight)) / 2.0
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        if inc_var_loss:
            inv_var = torch.exp(-logvar)
            mse_loss = torch.mean(torch.mean(torch.square(mean - labels) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.square(mean - labels), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def update(self, loss):
        self.optimizer.zero_grad()
        loss = loss + 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            loss = loss + self.get_decay_loss()
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel:
    def __init__(
        self,
        network_size,
        elite_size,
        state_size,
        action_size,
        reward_size=1,
        hidden_size=200,
        learning_rate=1e-3,
        use_decay=False,
        device=None,
    ):
        self.network_size = int(network_size)
        self.elite_size = int(elite_size)
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.reward_size = int(reward_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.elite_model_idxes = list(range(min(self.elite_size, self.network_size)))
        self.scaler = StandardScaler()
        self.ensemble_model = EnsembleModel(
            input_size=self.state_size + self.action_size,
            output_size=self.reward_size,
            ensemble_size=self.network_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            use_decay=use_decay,
            device=self.device,
        )
        self.last_train_stats = {}

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0.2, max_epochs_since_update=5):
        inputs = np.asarray(inputs, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        if inputs.shape[0] < 2:
            raise ValueError("Need at least two samples to train the ensemble reward model.")

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        num_holdout = min(max(1, num_holdout), inputs.shape[0] - 1)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]
        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs_t = torch.from_numpy(holdout_inputs).float().to(self.device)
        holdout_labels_t = torch.from_numpy(holdout_labels).float().to(self.device)
        holdout_inputs_t = holdout_inputs_t[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels_t = holdout_labels_t[None, :, :].repeat([self.network_size, 1, 1])

        snapshots = {i: (None, 1e10) for i in range(self.network_size)}
        epochs_since_update = 0
        last_holdout_losses = None
        final_epoch = 0

        for epoch in itertools.count():
            train_idx = np.vstack([
                np.random.permutation(train_inputs.shape[0])
                for _ in range(self.network_size)
            ])
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos:start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(self.device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(self.device)
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.update(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs_t, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(
                    holdout_mean,
                    holdout_logvar,
                    holdout_labels_t,
                    inc_var_loss=False,
                )
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                last_holdout_losses = holdout_mse_losses
                self.elite_model_idxes = np.argsort(holdout_mse_losses)[:self.elite_size].tolist()

            updated = False
            for i, current_loss in enumerate(holdout_mse_losses):
                _, best_loss = snapshots[i]
                improvement = (best_loss - current_loss) / best_loss
                if improvement > 0.01:
                    snapshots[i] = (epoch, current_loss)
                    updated = True
            epochs_since_update = 0 if updated else epochs_since_update + 1
            final_epoch = epoch
            if epochs_since_update > max_epochs_since_update:
                break

        self.last_train_stats = {
            "epochs": final_epoch + 1,
            "holdout_losses": last_holdout_losses,
            "elite_model_idxes": list(self.elite_model_idxes),
            "holdout_loss_mean": float(np.mean(last_holdout_losses)),
        }
        return self.last_train_stats

    def predict(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            batch = inputs[i:min(i + batch_size, inputs.shape[0])]
            batch_t = torch.from_numpy(batch).float().to(self.device)
            batch_t = batch_t[None, :, :].repeat([self.network_size, 1, 1])
            with torch.no_grad():
                batch_mean, batch_var = self.ensemble_model(batch_t, ret_log_var=False)
            ensemble_mean.append(batch_mean.detach().cpu().numpy())
            ensemble_var.append(batch_var.detach().cpu().numpy())

        ensemble_mean = np.concatenate(ensemble_mean, axis=1)
        ensemble_var = np.concatenate(ensemble_var, axis=1)
        if factored:
            return ensemble_mean, ensemble_var

        mean = np.mean(ensemble_mean, axis=0)
        var = np.mean(ensemble_var, axis=0) + np.mean(np.square(ensemble_mean - mean[None, :, :]), axis=0)
        return mean, var

    def predict_reward(self, inputs, deterministic=False):
        means, variances = self.predict(inputs, factored=True)
        batch_size = means.shape[1]
        if deterministic:
            model_idxes = np.asarray(self.elite_model_idxes)[
                np.arange(batch_size) % len(self.elite_model_idxes)
            ]
            return means[model_idxes, np.arange(batch_size), 0]

        stds = np.sqrt(np.maximum(variances, 1e-12))
        samples = means + np.random.normal(size=means.shape) * stds
        model_idxes = np.random.choice(self.elite_model_idxes, size=batch_size)
        return samples[model_idxes, np.arange(batch_size), 0]
