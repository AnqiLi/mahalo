import torch
from lightATAC.util import sample_batch, torchify


def evaluate_policy(env, policy, max_episode_steps, deterministic=True, discount = 0.99):
    """
    Modified from lightATAC.util.evaluate_policy
    """
    obs = env.reset()
    total_reward = 0.
    discount_total_reward = 0.
    for i in range(max_episode_steps):
        with torch.no_grad():
            try:
                action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
            except:
                action = policy.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        discount_total_reward += reward * discount ** i
        if done:
            break
        else:
            obs = next_obs
    # For Meta-World, log whether the episode is successful
    success = info.get('success', False)
    return [total_reward, discount_total_reward, float(success)]

def get_log_fun(writer=None, prefix=''):
    """
    Logs metrics into a tensorboard writer if writer is not None,
    otherwise prints the metrics
    """
    if writer is not None:
        def log_fun(metrics, step):
            print(metrics)
            for k, v in metrics.items():
                writer.add_scalar(prefix + k, v, step)
    else:
        log_fun = lambda metrics, step: print(metrics)
    return log_fun


def sample_join_batch(dataset, dataset_reward, batch_size):
    """
    Samples a batch from both dataset and dataset_reward
    """
    batch = sample_batch(dataset, batch_size)
    batch_reward = sample_batch(dataset_reward, batch_size)
    batch['observations_r'] = batch_reward['observations']
    batch['next_observations_r'] = batch_reward['next_observations']
    batch['rewards_r'] = batch_reward['rewards']
    batch['terminals_r'] = batch_reward['terminals']
    return batch
