import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightATAC.util import compute_batched, update_exponential_moving_average
from lightATAC.atac import normalized_sum, l2_projection

class MAHALO(nn.Module):
    """ Modified from lightATAC.atac.ATAC """
    def __init__(self, *,
                 policy,
                 qf,
                 rf,
                 optimizer,
                 discount=0.99,
                 Vmin=-float('inf'), # min value of Q (used in target backup)
                 Vmax=float('inf'), # max value of Q (used in target backup)
                 rmin = -float('inf'), # min value of reward (used in target backup)
                 rmax = float('inf'), # max value of reward (uesd in target backup)
                 action_shape,  # shape of the action space
                 # Optimization parameters
                 policy_lr=5e-7,
                 qf_lr=5e-4,
                 target_update_tau=5e-3,
                 fixed_alpha=None,
                 target_entropy=None,
                 initial_log_entropy=0.,
                 # ATAC parameters
                 beta=1.0,  # the regularization coefficient in front of the Bellman error
                 norm_constraint=100,  # l2 norm constraint on the NN weight
                 alpha_beta_ratio=100,
                 # PSPI parameters
                 init_observations=None, # Provide it to use PSPI (None or np.ndarray)
                 buffer_batch_size=256,  # for PSPI (sampling batch_size of init_observations)
                 ):

        #############################################################################################
        super().__init__()
        assert beta>=0 and norm_constraint>=0
        policy_lr = qf_lr if policy_lr is None or policy_lr < 0 else policy_lr # use shared lr if not provided.

        # MAHALO extra parameters
        self.alpha_beta_ratio = alpha_beta_ratio

        # ATAC main parameter
        self.beta = beta # regularization constant on the Bellman surrogate

        # q update parameters
        self._discount = discount
        self._Vmin = Vmin  # lower bound on the target
        self._Vmax = Vmax  # upper bound on the target
        self._rmin = rmin  # lower bound on reward
        self._rmax = rmax  # upper bound on reward
        self._norm_constraint = norm_constraint  # l2 norm constraint on the qf's weight; if negative, it gives the weight decay coefficient.

        # networks
        self.policy = policy
        self._qf = qf
        self._target_qf = copy.deepcopy(self._qf).requires_grad_(False)
        self._rf = rf

        # optimization
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._alpha_lr = qf_lr # potentially a larger stepsize, for the most inner optimization.
        self._tau = target_update_tau

        self._optimizer = optimizer
        self._policy_optimizer = self._optimizer(self.policy.parameters(), lr=self._policy_lr) #  lr for warmstart
        self._qf_optimizer = self._optimizer(list(self._qf.parameters()) + list(self._rf.parameters()), lr=self._qf_lr)

        # control of policy entropy
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            self._target_entropy = target_entropy if target_entropy else -np.prod(action_shape).item()
            self._log_alpha = torch.nn.Parameter(torch.Tensor([initial_log_entropy]))
            self._alpha_optimizer = optimizer([self._log_alpha], lr=self._alpha_lr)
        else:
            self._log_alpha = torch.Tensor([self._fixed_alpha]).log()

        # PSPI
        self._init_observations = torch.Tensor(init_observations) if init_observations is not None else init_observations  # if provided, it runs PSPI
        self._buffer_batch_size = buffer_batch_size


    def update(self, observations, actions, next_observations, terminals,
                     observations_r, next_observations_r, rewards_r, terminals_r, **kwargs):
        terminals = terminals.flatten().float()
        rewards = self._rf(observations, next_observations, terminals) # predict rewards

        ##### Update Critic #####
        def compute_bellman_backup(q_pred_next):
            assert rewards.shape == q_pred_next.shape
            q_pred_next = torch.clamp(q_pred_next, min=self._Vmin, max=self._Vmax)
            rewards_clip = torch.clamp(rewards, min=self._rmin, max=self._rmax)
            future_value = (1. - terminals) * self._discount * q_pred_next
            return rewards_clip.detach() + future_value, rewards + future_value.detach()

        # Pre-computation
        with torch.no_grad():  # regression target
            new_next_actions = self.policy(next_observations).rsample()
            target_q_values = self._target_qf(next_observations, new_next_actions)  # projection
        q_target, q_target_reward = compute_bellman_backup(target_q_values.flatten())

        new_actions_dist = self.policy(observations)  # This will be used to compute the entropy
        new_actions = new_actions_dist.rsample() # These samples will be used for the actor update too, so they need to be traced.

        if self._init_observations is None:  # ATAC
            pess_new_actions = new_actions.detach()
            pess_observations = observations
        else:  # PSPI
            idx_ = np.random.choice(len(self._init_observations), self._buffer_batch_size)
            init_observations = self._init_observations[idx_]
            init_actions_dist = self.policy(init_observations)
            pess_new_actions = init_actions_dist.rsample().detach()
            pess_observations = init_observations

        qf_pred_both, qf_pred_next_both, qf_new_actions_both \
            = compute_batched(self._qf.both, [observations, next_observations, pess_observations],
                                             [actions,      new_next_actions,  pess_new_actions])

        pess_loss = 0
        qf_bellman_loss = 0
        rf_bellman_loss = 0
        w1, w2 = 0.5, 0.5
        for qfp, qfpn, qfna in zip(qf_pred_both, qf_pred_next_both, qf_new_actions_both):
            # Compute Bellman error
            assert qfp.shape == qfpn.shape == qfna.shape == q_target.shape
            target_error_qf = F.mse_loss(qfp, q_target)
            target_error_rf = F.mse_loss(qfp.detach(), q_target_reward)
            q_target_pred, q_target_pred_reward = compute_bellman_backup(qfpn)
            td_error_qf = F.mse_loss(qfp, q_target_pred)
            td_error_rf = F.mse_loss(qfp.detach(), q_target_pred_reward)
            qf_bellman_loss += w1 * target_error_qf + w2 * td_error_qf
            rf_bellman_loss += w1 * target_error_rf + w2 * td_error_rf
            # Compute pessimism term
            if self._init_observations is None:  # ATAC
                _pess_loss = (qfna - qfp).mean()
                pess_loss += _pess_loss
            else:  # PSPI
                _pess_loss = qfna.mean()
                pess_loss += _pess_loss

        # Reward loss
        rf_pred = self._rf(observations_r, next_observations_r, terminals_r)
        rf_loss = F.mse_loss(rf_pred, rewards_r)


        qf_loss = normalized_sum(pess_loss, qf_bellman_loss, self.beta) + \
                  normalized_sum(rf_bellman_loss, rf_loss, self.alpha_beta_ratio)

        # Update qf and rf
        self._qf_optimizer.zero_grad()
        qf_loss.backward()
        self._qf_optimizer.step()
        self._qf.apply(l2_projection(self._norm_constraint))
        update_exponential_moving_average(self._target_qf, self._qf, self._tau)

        ##### Update Actor #####
        # Compuate entropy
        log_pi_new_actions = new_actions_dist.log_prob(new_actions)
        policy_entropy = -log_pi_new_actions.mean()

        alpha_loss = 0
        if self._use_automatic_entropy_tuning:  # it comes first; seems to work also when put after policy update
            alpha_loss = self._log_alpha * (policy_entropy.detach() - self._target_entropy)  # entropy - target
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        # Compute performance difference lower bound (policy_loss = - lower_bound - alpha * policy_kl)
        alpha = self._log_alpha.exp().detach()
        self._qf.requires_grad_(False)
        lower_bound = self._qf.both(observations, new_actions)[-1].mean() # just use one network
        self._qf.requires_grad_(True)
        policy_loss = normalized_sum(-lower_bound, -policy_entropy, alpha)

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # Logging
        with torch.no_grad():
            log_info = dict(policy_loss=policy_loss.item(),
                            qf_loss=qf_loss.item(),
                            qf_bellman_loss=qf_bellman_loss.item(),
                            rf_bellman_loss=rf_bellman_loss.item(),
                            pess_loss=pess_loss.item(),
                            rf_loss=rf_loss.item(),
                            alpha_loss=alpha_loss.item(),
                            policy_entropy=policy_entropy.item(),
                            alpha=alpha.item(),
                            lower_bound=lower_bound.item(),
                            rf_pred_reward_batch_mean = rf_pred.mean().item(), # reward prediction on reward dataset
                            rf_pred_action_batch_mean = rewards.mean().item(), # reward prediction on dynamics dataset
                            qf_bellman_surrogate=td_error_qf.item(),
                            rf_bellman_surrogate=td_error_rf.item(),
                            qf1_pred_mean=qf_pred_both[0].mean().item(),
                            qf2_pred_mean = qf_pred_both[1].mean().item(),
                            q_target_mean = q_target.mean().item(),
                            target_q_values_mean = target_q_values.mean().item(),
                            qf1_new_actions_mean = qf_new_actions_both[0].mean().item(),
                            qf2_new_actions_mean = qf_new_actions_both[1].mean().item(),
                            action_diff = torch.mean(torch.norm(actions - new_actions, dim=1)).item())

        return log_info
