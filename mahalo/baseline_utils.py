import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from lightATAC.util import mlp, torchify, sample_batch
from tqdm import trange
import copy

class InverseDynamicsFunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2, use_tanh=True):
        super().__init__()
        dims = [state_dim * 2 + 1, *([hidden_dim] * n_hidden), action_dim]
        self.idf = mlp(dims, output_activation=nn.Tanh if use_tanh else None)

    def forward(self, state, next_state, terminal):
        """
        Predicts the action given state transitions
        """
        state_transition = torch.cat([state, next_state, terminal.unsqueeze(1)], 1)
        return self.idf(state_transition)


def train_mse(dataset, predictor, *,
              key, n_steps, lr, batch_size=256,
              log_freq=1000, log_fun=None, silence=False):
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    for step in trange(n_steps, disable=silence):
        batch = sample_batch(dataset, batch_size)
        terminals = batch['terminals'].flatten().float()
        preds = predictor(batch['observations'], batch['next_observations'], terminals)
        loss = F.mse_loss(preds, batch[key])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_metrics = {f"{key} train mean": preds.mean().item(),
                         f"{key} train std": preds.std().item(),
                         f"{key} train max": preds.max().item(),
                         f"{key} train min": preds.min().item(),
                         f"{key} loss": loss.item()}
        if (step % max(log_freq,1) == 0 or step==n_steps-1) and log_fun is not None:
            log_fun(train_metrics, step)


def label_dataset(dataset, predictor, key, batch_size, vmin=None, vmax=None, log_fun=None):
    dataset_labeled = copy.deepcopy(dataset)
    n = len(dataset['observations'])
    n_batch = int(np.ceil(n / batch_size))
    log_freq = int(n_batch / 100)
    predictions = []
    with torch.no_grad():
        for i in trange(n_batch):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n)
            observations = torchify(dataset['observations'][start_idx:end_idx])
            next_observations = torchify(dataset['next_observations'][start_idx:end_idx])
            terminals = torchify(dataset['terminals'][start_idx:end_idx]).flatten().float()
            preds = predictor(observations, next_observations, terminals).cpu().numpy()
            predictions.append(preds)
            if (i % max(log_freq,1) == 0 or i==n_batch-1) and log_fun is not None:
                label_metrics = {f"{key} label mean": preds.mean(),
                                 f"{key} label std": preds.std(),
                                 f"{key} label max": preds.max(),
                                 f"{key} label min": preds.min()}
                log_fun(label_metrics, i)
        dataset_labeled[key] = np.clip(np.concatenate(predictions),
                                       a_min=vmin, a_max=vmax)
    return dataset_labeled
