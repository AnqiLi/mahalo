from pathlib import Path
import gym, d4rl
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from lightATAC.atac import ATAC
from lightATAC.policy import GaussianPolicy
from lightATAC.value_functions import TwinQ
from lightATAC.util import Log, set_seed
from lightATAC.util import (sample_batch, DEFAULT_DEVICE)
from lightATAC.bp import BehaviorPretraining
from mahalo.reward_functions import StateTransitionRewardFunction
from mahalo.utils import get_log_fun, evaluate_policy
from mahalo.env_utils import get_data, concatenate_datasets, get_benchmark
from mahalo.baseline_utils import InverseDynamicsFunction, train_mse, label_dataset


EPS=1e-6

pretrain_log_freq = 1000
disable_tqdm = True

def eval_agent(*, env, agent, discount, n_eval_episodes, max_episode_steps=1000,
               deterministic_eval=True, normalize_score=None):

    all_returns = np.array([evaluate_policy(env, agent, max_episode_steps, deterministic_eval, discount) \
                             for _ in range(n_eval_episodes)])
    eval_returns = all_returns[:,0]
    discount_returns = all_returns[:,1]
    success = all_returns[:,2]

    info_dict = {
        "return mean": eval_returns.mean(),
        "return std": eval_returns.std(),
        "discounted returns": discount_returns.mean(),
        "success rate": success.mean()
    }

    if normalize_score is not None:
        normalized_returns = normalize_score(eval_returns)
        info_dict["normalized return mean"] = normalized_returns.mean()
        info_dict["normalized return std"] =  normalized_returns.std()
    return info_dict


def main(args):
    if args.algo in ["uds", "uds-a", "common"] and args.scenario in ['ilfo', 'rlfo']:
        print(f"Invalid algorithm: {args.algo} can not be used in LFO setting.")
        return

    # ------------------ Initialization ------------------ #
    torch.set_num_threads(1)

    set_seed(args.seed)

    data = get_data(args.env,
                    scenario=args.scenario,
                    reward_dataset_ratio=args.reward_dataset_ratio,
                    remove_info=False,
                    uds=(args.algo=='uds'),
                    remove_terminals=args.remove_terminals)

    env = data['env']
    env_id = data['env_id']
    dataset_action = data['dataset']
    dataset_reward = data['dataset_reward']

    set_seed(args.seed, env=env)

    # Setup logger
    exp_name = args.env + '-' + args.scenario + f"-{args.algo}"
    log_path = Path(args.log_dir) / exp_name / ('_beta' + str(args.beta))
    # Log reward dataset shape
    args.dataset_reward_size = dataset_reward['rewards'].shape[0]
    log = Log(log_path, vars(args))
    log(f'Log dir: {log.dir}')
    writer = SummaryWriter(log.dir)

    # Assume vector observation and action
    obs_dim, act_dim = dataset_action['observations'].shape[1], dataset_action['actions'].shape[1]

    if 'il' in args.scenario:
        # For imitation learning, bounds reward prediction to [0, 1]
        r_min = 0.
        r_max = 1.
    else:
        # For RL, use max and min reward from data
        r_min = np.min(dataset_reward['rewards'])
        r_max = np.max(dataset_reward['rewards'])

    V_min = min(-1.0 / (1 - args.discount), 2 * r_min / (1 - args.discount)) if args.clip_v else -float('inf')
    V_max = max( 1.0 / (1 - args.discount), 2 * r_max / (1 - args.discount)) if args.clip_v else float('inf')

    rf = StateTransitionRewardFunction(obs_dim,
                                       hidden_dim=args.hidden_dim,
                                       n_hidden=args.n_hidden,
                                       ).to(DEFAULT_DEVICE)
    idf = InverseDynamicsFunction(obs_dim, act_dim,
                                  hidden_dim=args.hidden_dim,
                                  n_hidden=args.n_hidden).to(DEFAULT_DEVICE)
    qf = TwinQ(obs_dim, act_dim,
               hidden_dim=args.hidden_dim,
               n_hidden=args.n_hidden).to(DEFAULT_DEVICE)
    policy = GaussianPolicy(obs_dim, act_dim,
                            hidden_dim=args.hidden_dim,
                            n_hidden=args.n_hidden,
                            use_tanh=True,
                            std_type='diagonal').to(DEFAULT_DEVICE)

    rl = ATAC(policy=policy,
              qf=qf,
              optimizer=torch.optim.Adam,
              discount=args.discount,
              action_shape=act_dim,
              buffer_batch_size=args.batch_size,
              policy_lr=args.slow_lr,
              qf_lr=args.fast_lr,
              Vmin=V_min,
              Vmax=V_max,
              # ATAC main parameters
              beta=args.beta) # the regularization coefficient in front of the Bellman error
    rl.to(DEFAULT_DEVICE)

    # ------------------ Pretraining ------------------ #
    pretrain_model_dir = Path(args.log_dir) / 'pretrain' / exp_name
    pretrain_model_path = pretrain_model_dir / 'model.pt'
    # --------- Pretraining reward function --------- #
    print("Pretraining")
    if args.algo in ['rp', 'arp']:
        print("Training reward function")
        reward_log_fun = get_log_fun(writer, prefix="Rewards/")
        # Learn a reward function
        train_mse(dataset_reward,
                    rf,
                    key='rewards',
                    n_steps=args.n_warmstart_steps,
                    lr=args.fast_lr,
                    batch_size=args.batch_size,
                    log_fun=reward_log_fun,
                    log_freq=pretrain_log_freq,
                    silence=disable_tqdm)

        # Label action dataset with rewards
        del dataset_action['rewards']
        dataset_action = label_dataset(dataset_action, rf,
                                        key='rewards',
                                        batch_size=args.batch_size,
                                        vmin=r_min, vmax=r_max,
                                        log_fun=reward_log_fun)

    if args.algo in ['ap', 'arp']:
        print("Training inverse dynamics")
        action_log_fun = get_log_fun(writer, prefix="Actions/")
        # Learn an inverse dynamics function
        del dataset_reward['actions']
        train_mse(dataset_action,
                    idf,
                    key='actions',
                    n_steps=args.n_warmstart_steps,
                    lr=args.fast_lr,
                    batch_size=args.batch_size,
                    log_fun=action_log_fun,
                    log_freq=pretrain_log_freq,
                    silence=disable_tqdm)
        # # Label reward dataset with actions
        dataset_reward = label_dataset(dataset_reward, idf,
                                        key='actions',
                                        batch_size=args.batch_size,
                                        vmin=-1.+EPS, vmax=1.-EPS,
                                        log_fun=action_log_fun)

    if args.algo == 'rp':
        dataset = dataset_action
    elif args.algo == 'ap':
        dataset = dataset_reward
    elif args.algo == 'arp':
        dataset = concatenate_datasets(dataset_action, dataset_reward)
    elif args.algo in ['uds', 'uds-a']:
        dataset_action['rewards'] = r_min * np.ones_like(dataset_action['observations'][:, 0])
        dataset = concatenate_datasets(dataset_action, dataset_reward)
    elif args.algo == 'common':
        # Assuming dataset_reward is contained in dataset_action
        dataset = dataset_reward
    elif args.algo == 'oracle':
        # Assumes dataset action all has reward
        dataset = dataset_action
    else:
        raise ValueError("Unknown algorithm.")
    dataset['actions'] = np.clip(dataset['actions'], -1+EPS, 1-EPS)  # due to tanh

    pretrain_log_fun = get_log_fun(writer=writer,
                                    prefix="Pretraining/")
    print("Training policy and value functions")
    # Trains policy and value to fit the behavior data from the dynamics dataset
    pt = BehaviorPretraining(qf=qf,
                                policy=policy,
                                lr=args.fast_lr, discount=args.discount,
                                Vmin=V_min,
                                Vmax=V_max,
                                td_weight=0.5, rs_weight=0.5,
                                fixed_alpha=None,
                                action_shape=act_dim).to(DEFAULT_DEVICE)
    pt.train(dataset,
                n_steps=args.n_warmstart_steps,
                batch_size=args.batch_size,
                log_fun=pretrain_log_fun,
                log_freq=pretrain_log_freq,
                silence=disable_tqdm)
    rl._target_qf = pt.target_qf
    # Saves pretrained model
    pretrain_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rl.state_dict(), pretrain_model_path)

    # Main Training
    # Logging
    train_log_fun = get_log_fun(writer=writer, prefix="Train/")
    eval_log_fun = get_log_fun(writer=writer, prefix="Eval/")
    if get_benchmark(env_id) == 'd4rl':
        normalize_score = lambda returns: d4rl.get_normalized_score(env_id, returns)*100.0
        max_episode_steps = 1000
    else:
        normalize_score = None
        max_episode_steps = 128
    for step in trange(args.n_steps, disable=disable_tqdm):
        # Evaluates the policy right after BC. This makes sure
        # that BCO policy performance is logged
        if step == 0:
            eval_metrics = eval_agent(env=env,
                                      agent=policy,
                                      discount=args.discount,
                                      max_episode_steps=max_episode_steps,
                                      n_eval_episodes=args.n_eval_episodes,
                                      normalize_score=normalize_score)
            log.row(eval_metrics)
            eval_log_fun(eval_metrics, step)
        # Samples a minibatch from each dataset and combine them
        train_metrics = rl.update(**sample_batch(dataset, args.batch_size))
        if step % max(int(args.eval_period/10),1) == 0 or step == (args.n_steps - 1):
            train_log_fun(train_metrics, step)
        if (step+1) % args.eval_period == 0 or step == (args.n_steps - 1):
            eval_metrics = eval_agent(env=env,
                                      agent=policy,
                                      discount=args.discount,
                                      max_episode_steps=max_episode_steps,
                                      n_eval_episodes=args.n_eval_episodes,
                                      normalize_score=normalize_score)
            log.row(eval_metrics)
            eval_log_fun(eval_metrics, step)
    # Final processing
    torch.save(rl.state_dict(), log.dir/'final.pt')
    log.close()
    writer.close()
    return eval_metrics.get('normalized return mean')


def get_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True) # rp, ap, arp, uds, uds-a, common, oracle
    parser.add_argument('--scenario', type=str, required=True) # il, ilfo, rl_expert, rlfo, rl_sample
    parser.add_argument('--reward_dataset_ratio', type=float, default=0.01)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=3)
    parser.add_argument('--n_steps', type=int, default=10**6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--fast_lr', type=float, default=5e-4)
    parser.add_argument('--slow_lr', type=float, default=5e-7)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--eval_period', type=int, default=50000)
    parser.add_argument('--n_eval_episodes', type=int, default=50)
    parser.add_argument('--n_warmstart_steps', type=int, default=100*10**3)
    parser.add_argument('--remove_terminals', action='store_true')
    parser.add_argument('--clip_v', action='store_true')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
