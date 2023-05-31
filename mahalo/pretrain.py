import torch
from tqdm import trange
from torch.nn import functional as F
from lightATAC.util import tuple_to_traj_data, traj_data_to_qlearning_data
from lightATAC.bp import BehaviorPretraining
from mahalo.utils import sample_join_batch

class Pretraining(BehaviorPretraining):
    def __init__(self, *,
                 rf,  # nn.module
                 policy=None,  # nn.module
                 qf=None, vf=None,  # nn.module
                 discount=0.99,  # discount factor
                 Vmin=-float('inf'), # min value of Q (used in target backup)
                 Vmax=float('inf'), # max value of Q (used in target backup)
                 lr=5e-4,  # learning rate
                 use_cache=False,
                 # Q learning
                 target_update_rate=0.005,
                 td_weight=1.0,  # td error weight (surrogate based on target network)
                 rs_weight=0.0,  # residual error weight
                 # V learning
                 lambd=1.0, # lambda for TD lambda
                 expectile=0.5, # expectile for expectile regress, MSE if expectile=0.5
                 # Policy entropy
                 action_shape=None,
                 fixed_alpha=0.,
                 target_entropy=None,
                 initial_log_alpha=0.,
                 # R learning
                 rmin=-float('inf'), # min reward
                 rmax=float('inf'), # max reward
                 alpha_beta_ratio=1,  # weight for reward loss
                 ):
        """
            Inputs:
                rf: An nn.module of reward network
                policy: An nn.module of policy network that returns a distribution or a tensor.
                qf: An nn.module of double Q networks that implement additionally `both` method.
                vf: An nn.module of value network.

                Any provided networks above would be trained; to train `qf`, `policy` is required.

                discount: discount factor
                Vmin: min value of Q (used in target backup)
                Vmax: max value of Q (used in target backup)
                lr: learning rate
                use_cache: whether to batch compute the policy and cache the result
                target_update_rate: learning rate to update target network
                td_weight: weight on the TD loss for learning `qf`
                rs_weight: weight on the residual loss for learning `qf`
                lambd: lambda in TD-lambda for learning `vf`.
                expectile: expectile used for learning `vf`.

                action_shape: shape of the vector action space
                fixed_alpha: weight on the entropy term
                target_entropy: entropy lower bound; if None, it would be inferred from `action_shape`
                initial_log_alpha: initial log entropy

                rmin: min reward
                rmax: max reward
                alpha_beta_ratio: weight of reward loss
        """
        super().__init__(policy=policy,
                         qf=qf, vf=vf,
                         discount=discount,
                         Vmin=Vmin,
                         Vmax=Vmax,
                         lr=lr,
                         use_cache=use_cache,
                         target_update_rate=target_update_rate,
                         td_weight=td_weight,
                         rs_weight=rs_weight,
                         lambd=lambd,
                         expectile=expectile,
                         action_shape=action_shape,
                         fixed_alpha=fixed_alpha,
                         target_entropy=target_entropy,
                         initial_log_alpha=initial_log_alpha)

        self.rf = rf
        self.rf_optimizer = torch.optim.Adam(rf.parameters(), lr=lr)
        self.alpha_beta_ratio = alpha_beta_ratio
        self._rmin = rmin
        self._rmax = rmax

    def train(self, dataset, dataset_reward, n_steps, batch_size=256, log_freq=1000, log_fun=None, silence=False):
        """ A basic trainer loop.
            dataset: a dict of observations, actions, next observations
            dataset_reward: a dict of observations, next observations, rewards, terminals
            # TODO: potentially learn terminals along with rewards
        """
        if self.vf is not None:
            traj_data = tuple_to_traj_data(dataset)
            self.preprocess_traj_data(traj_data, self.discount)
            dataset = traj_data_to_qlearning_data(traj_data)  # make sure `next_observations` is there

        for step in trange(n_steps, disable=silence):
            batch = sample_join_batch(dataset, dataset_reward, batch_size)
            # label rewards
            # TODO: Here, for simplicity, we clamp reward directly.
            # The reward gradient from qf loss can be lost due to this clamping.
            # Potentially needs to rewrite compute_q_loss like in Mahalo
            batch['rewards'] = torch.clamp(self.rf(batch['observations'],
                                                   batch['next_observations'],
                                                   batch['terminals']), min=self._rmin, max=self._rmax)
            train_metrics = self.update(**batch)
            if (step % max(log_freq,1) == 0 or step==n_steps-1) and log_fun is not None:
                log_fun(train_metrics, step)

    def update(self, **batch):
        self.rf_optimizer.zero_grad()
        # Update policy, vf, qf
        info_dict = super().update(**batch)
        # Update rf
        rf_loss, rf_info_dict = self.compute_rf_loss(**batch)
        rf_info_dict['Reward_average_value_action_batch'] = batch['rewards'].mean().item()
        loss = self.alpha_beta_ratio * rf_loss
        loss.backward()
        self.rf_optimizer.step()
        # Log
        info_dict = {**info_dict, **rf_info_dict}
        return info_dict

    def compute_rf_loss(self, observations_r, next_observations_r, rewards_r, terminals_r, **kwargs):
        terminals_r = terminals_r.flatten().float()
        rs_pred = self.rf(observations_r, next_observations_r, terminals_r)
        reward_loss = F.mse_loss(rs_pred, rewards_r)
        return reward_loss, {"Reward_loss": reward_loss.item(),
                             "Reward_average_value_reward_batch": rs_pred.mean().item()}
