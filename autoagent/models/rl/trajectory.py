import torch
import numpy as np


class ReplayMemory:
    """
    Standard replay buffer
    """
    def __init__(self, size, num_features, action_dim):
        self.size = int(size)
        self.samples_added = 0
        self.num_features = int(num_features)
        self.action_dim = int(action_dim)
        self.curr_index = 0

        self.states = np.empty((self.size, self.num_features), dtype=np.float32)
        self.actions = np.empty((self.size, self.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.size, 1), dtype=np.float32)
        self.next_states = np.empty((self.size, self.num_features), dtype=np.float32)
        self.dones = np.empty((self.size, 1), dtype=np.float32)

    def add_sample(self, sample):
        s, a, r, ns, d = sample
        self.states[self.curr_index] = s
        self.actions[self.curr_index] = a
        self.rewards[self.curr_index] = r
        self.next_states[self.curr_index] = ns
        self.dones[self.curr_index] = d

        self.samples_added += 1
        self.curr_index = (self.curr_index + 1) % self.size

    def sample_batch(self, batch_size):
        random_indices = np.random.randint(min(self.samples_added, self.size), size=batch_size)
        return (
            self.states[random_indices],
            self.actions[random_indices],
            self.rewards[random_indices],
            self.next_states[random_indices],
            self.dones[random_indices],
        )


def gae(gamma, lambd, vfuncs, states, rewards, boot_value):
    """
    Generalized Advantage Estimation
    Computes trajectory targets and advantages using
    generalized advantage estimation (https://arxiv.org/abs/1506.02438)

    Parameters
    ----------
    gamma : float
        Discount factor
    lambd : float
        GAE lambda
    vfuncs : np.array
        Value function estimates for trajectory states
    states : np.array
        Trajectory states
    rewards : np.array
        Trajectory rewards
    boot_value : float
        Bootstrap value for the last state

    Returns
    -------
    Tuple[np.array, np.array]
        Trajectory targets and advantages
    """
    traj_len = states.shape[0]

    # Compute targets
    targets = np.zeros((traj_len), dtype=np.float32)
    curr_target = boot_value
    for i in reversed(range(traj_len)):
        targets[i] = rewards[i] + gamma * curr_target
        curr_target = targets[i]

    # Compute advantages
    advantages = np.zeros((traj_len), dtype=np.float32)
    curr_advantage = 0
    for i in reversed(range(traj_len)):
        v_next = boot_value if i == traj_len - 1 else vfuncs[i+1]
        advantages[i] = ((rewards[i] + gamma * v_next - vfuncs[i])
                         + gamma * lambd * curr_advantage)
        curr_advantage = advantages[i]

    return targets, advantages


def collect_trajectories_with_gae(vec_env, act, vfunc, gamma, lambd, num_samples,
                                  ob_shape, ac_shape):
    """
    Collects sample trajectories, computing targets and advantages using GAE.

    Parameters
    ----------
    vec_env : (gym.vector.AsyncVectorEnv | gym.vector.SyncVectorEnv)
        Vectorized gym-like environment
    act : Callable
        Callable that drives the action selection
    vfunc : Callable
        Callable to call the value-function on a state
    gamma : float
        Discount factor
    lambd : float
        Gae lambda
    num_samples : int
        The total number of samples (trajectory steps) to collect
    ob_shape : int
        Observation shape, should be passed as np.prod(observation_shape)
    ac_shape : int
        Action shape

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array, np.array]
        Trajectory states, actions, rewards, targets, advantages
    """
    num_envs = vec_env.num_envs
    states = np.zeros((num_envs, num_samples, ob_shape), dtype=np.float32)
    actions = np.zeros((num_envs, num_samples, ac_shape), dtype=np.float32)
    rewards = np.zeros((num_envs, num_samples), dtype=np.float32)
    targets = np.zeros((num_envs, num_samples), dtype=np.float32)
    advantages = np.zeros((num_envs, num_samples), dtype=np.float32)
    starts = np.zeros((num_envs), dtype=np.int32)
    count = 0

    s = vec_env.reset()
    while count < num_samples:
        states[:, count] = s.reshape(*states[:, count].shape)

        a = act(s)
        actions[:, count] = a.reshape(*actions[:, count].shape)

        # NOTE: envs where d is True are automatically reset by the gym vector wrapper
        ns, r, d, info = vec_env.step(a)

        # Real dones (not due to time limits)
        real_d = [d[i] and not info[i].get('TimeLimit.truncated', False) for i in range(num_envs)]

        rewards[:, count] = r
        count += 1

        # Augmented dones, here done can be True if
        # - environment done is True (either because of a real end or a time limit)
        # - we collected enough samples (count == num_samples)
        aug_d = [i for i in range(num_envs) if d[i] or (count==num_samples)]

        if aug_d:
            # We have atleast one complete trajectory to process
            vfuncs = [vfunc(states[env_idx, starts[env_idx]:count]) for env_idx in aug_d]
            boot_values = [
                # Boostrap in case the done is due to a time limit
                0 if real_d[env_idx] else vfunc(ns[env_idx])
                for env_idx in aug_d
            ]
            gae_traj = [
                gae(gamma, lambd, vfuncs[i], states[env_idx, starts[env_idx]:count],
                    rewards[env_idx, starts[env_idx]:count], boot_values[i])
                for i, env_idx in enumerate(aug_d)
            ]
            for i, env_idx in enumerate(aug_d):
                targets[env_idx, starts[env_idx]:count] = gae_traj[i][0]
                advantages[env_idx, starts[env_idx]:count] = gae_traj[i][1]

        # Update starts
        starts[aug_d] = count

        s = ns

    return states, actions, rewards, targets, advantages
