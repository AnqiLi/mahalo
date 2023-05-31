import d4rl, gym
from d4rl.infos import DATASET_URLS
from lightATAC.util import tuple_to_traj_data, traj_data_to_qlearning_data
import numpy as np
import copy
from urllib.error import HTTPError


def get_benchmark(env_id):
    """

    """
    env_name = env_id.split('-')[0]
    if env_name in ['hopper', 'walker2d', 'halfcheetah']:
        # D4RL dataset
        return 'd4rl'
    else:
        raise NotImplementedError("Environment not implemented")


def get_d4rl_dataset(env_id):
    env = gym.make(env_id) # d4rl ENV
    while True:
        try:
            dataset = env.get_dataset()
            # This ensures "next_observations" is in `dataset`.
            # Also removes metadata from keys
            dataset = traj_data_to_qlearning_data(tuple_to_traj_data(dataset))
            return dataset
        except (HTTPError, OSError):
            print('Unable to download dataset. Retry.')


def get_dataset(env_id):
    if get_benchmark(env_id) == 'd4rl':
        # D4RL dataset
        return get_d4rl_dataset(env_id)
    else:
        raise NotImplementedError("Meta-World dataset and environment not released yet.")

def get_env(env_id):
    if get_benchmark(env_id) == 'd4rl':
        return gym.make(env_id)
    else:
        raise NotImplementedError("Meta-World dataset and environment not released yet.")


def concatenate_datasets(dataset1, dataset2, update_dataset1=True):
    if update_dataset1:
        dataset = dataset1
    else:
        dataset = copy.deepcopy(dataset1)
    for k in dataset.keys():
        assert isinstance(dataset[k], np.ndarray) and isinstance(dataset2[k], np.ndarray), \
                "Dataset contains values that are not np.ndarray. Consider adding it to ignores in `tuple_to_traj_data`."
        assert len(dataset[k].shape) == 1 or dataset[k].shape[1] == dataset2[k].shape[1], \
                "Dimension mismatch! Check if the environments have the same state and action space!"
        dataset[k] = np.concatenate((dataset[k], dataset2[k]))
    return dataset


def get_concatenated_dataset(env_id_list):
    dataset = None
    for env_id in env_id_list:
        sub_dataset = get_dataset(env_id)
        if dataset is None:
            dataset = copy.deepcopy(sub_dataset)
        else:
            dataset = concatenate_datasets(dataset, sub_dataset)
    return dataset


def get_env_id_list(env_id):
    if get_benchmark(env_id) == 'd4rl':
        return get_d4rl_env_id_list(env_id)
    elif get_benchmark(env_id) == 'mw':
        return get_mw_env_id_list(env_id)
    else:
        raise ValueError

def get_expert_env_id(env_id):
    if get_benchmark(env_id) == 'd4rl':
        return get_d4rl_expert_env_id(env_id)
    elif get_benchmark(env_id) == 'mw':
        return get_mw_expert_env_id(env_id)
    else:
        raise ValueError

def get_d4rl_env_id_list(env_id):
    if env_id in DATASET_URLS:
        # Single d4rl environment
        return [env_id]
    # Otherwise, all non-expert d4rl environment
    # with the same base env and version

    # Find the env name and version
    env_id_split = env_id.split('-')
    idx = -1
    try:
        idx = env_id.index('-all-')
    except ValueError:
        raise ValueError("Env id should either be from d4rl or contains `all` and version.")
    env_name = env_id[:idx]
    env_version = env_id_split[-1]

    # Find all d4rl environment with the same base env and version
    # Except the one contains `expert`
    env_list = []
    for d4rl_env_id in DATASET_URLS.keys():
        if env_name in d4rl_env_id and env_version in d4rl_env_id:
            if 'expert' not in d4rl_env_id:
                env_list.append(d4rl_env_id)
    return env_list

def get_d4rl_expert_env_id(env_id):
    # Find the base env and version
    env_id_split = env_id.split('-')
    idx = -1
    for i, s in enumerate(env_id_split):
        if s in ['all', 'medium', 'random', 'full', 'expert']:
            idx = i
            break
    assert idx > -1 and idx < len(env_id_split), "Non-standard env id."
    env_name = '-'.join(env_id_split[:idx])
    env_version = env_id_split[-1]
    expert_env_id = env_name + '-expert-' + env_version
    return expert_env_id

def get_mw_env_id_list(env_id):
    return [env_id + f'-noise{level}' for level in ['0.1', '0.5', '1']]

def get_mw_expert_env_id(env_id):
    return env_id + '-noise0'

def subsample_dataset(dataset, ratio):
    """
    Randomly sample trajectories from a dataset
    """
    # Chunk dataset into trajectories
    traj_data = tuple_to_traj_data(dataset)
    # Number of transitions in the sampled dataset
    sampled_dataset_size = int(len(dataset['observations']) * ratio)
    # Randomly samples trajectories
    permutation = np.random.permutation(np.arange(len(traj_data)))
    sampled_traj_data = []
    current_size = 0
    for idx in permutation:
        if len(traj_data[idx]['rewards']) == 1:
            # Ignores length-1 trajectories, which are likely artifacts
            # of hindsight labeling of terminals
            continue
        sampled_traj_data.append(copy.deepcopy(traj_data[idx]))
        current_size += len(traj_data[idx]['observations'])
        if current_size >= sampled_dataset_size:
            break
    # Converts to d4rl dataset form
    sampled_dataset = traj_data_to_qlearning_data(sampled_traj_data)
    return sampled_dataset

def get_data(env_id, *,
             scenario,
             reward_dataset_ratio=0.01,
             remove_info=True, uds=False,
             remove_terminals=False):
    if scenario == 'il':
        use_expert = True
        il = True
        lfo = False
    elif scenario == 'ilfo':
        use_expert = True
        il = True
        lfo = True
    elif scenario == 'rl_expert':
        use_expert = True
        il = False
        lfo = False
    elif scenario == 'rlfo':
        use_expert = True
        il = False
        lfo = True
    elif scenario == 'rl_sample':
        use_expert = False
        il = False
        lfo = False
    else:
        raise ValueError

    env_id_list = get_env_id_list(env_id)
    # Gets (reward-free) action dataset
    action_dataset = get_concatenated_dataset(env_id_list)

    # Gets reward dataset
    if use_expert:
        # Uses expert environment
        expert_env_id = get_expert_env_id(env_id)
        reward_dataset_full = get_dataset(expert_env_id)
    else:
        # Uses action dataset
        reward_dataset_full = copy.deepcopy(action_dataset)
    # Subsamples reward dataset
    reward_dataset = subsample_dataset(reward_dataset_full, reward_dataset_ratio)

    if remove_terminals:
        # Removes terminal transitions in action dataset
        traj_data = tuple_to_traj_data(action_dataset)
        for traj in traj_data:
            if traj['terminals'][-1]:
                for k in traj:
                    traj[k] = traj[k][:-1]
                traj['timeouts'][-1] = True
        action_dataset = traj_data_to_qlearning_data(traj_data)

    if use_expert and not lfo and not uds:
        # If not uds,
        # Adds expert dataset into action dataset
        extra_action_dataset = copy.deepcopy(reward_dataset)
        action_dataset = concatenate_datasets(action_dataset, extra_action_dataset)

    if il:
        reward_dataset['rewards'][:] = 1.

    if remove_info:
        del action_dataset['rewards']
        if not uds:
            # For uds loader, reward dataset contains action
            del reward_dataset['actions']

    # Makes gym environment
    env = get_env(env_id_list[0])
    return dict(env=env,
                env_id=env_id_list[0],
                dataset=action_dataset,
                env_id_list=[env_id] if (not use_expert or lfo) else [env_id, expert_env_id], # env_id_list,
                dataset_reward=reward_dataset,
                env_id_reward_list=[expert_env_id] if use_expert else env_id_list)
