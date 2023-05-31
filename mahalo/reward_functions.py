import torch
from torch import nn
from lightATAC.util import mlp
from abc import abstractclassmethod

class RewardFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2, output_activation=None):
        super().__init__()
        self.rf = self._build(state_dim, hidden_dim, n_hidden, output_activation)

    @abstractclassmethod
    def forward(self, state, next_state, terminal):
        """
        Predicts the reward function given state transitions
        """

    @abstractclassmethod
    def _build(self, state_dim, hidden_dim, n_hidden):
        """
        Returns reward network given specs
        """

class StateRewardFunction(RewardFunction):
    def forward(self, state, next_state, terminal):
        return self.rf(state)

    def _build(self, state_dim, hidden_dim, n_hidden, output_activation):
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        return mlp(dims, squeeze_output=True, output_activation=output_activation)

class StateTransitionRewardFunction(RewardFunction):
    def forward(self, state, next_state, terminal):
        state_transition = torch.cat([state, next_state, terminal.unsqueeze(1)], 1)
        return self.rf(state_transition)

    def _build(self, state_dim, hidden_dim, n_hidden, output_activation):
        dims = [state_dim * 2 + 1, *([hidden_dim] * n_hidden), 1]
        return mlp(dims, squeeze_output=True, output_activation=output_activation)