"""Off-policy buffer."""
import numpy as np
import torch
from harl.utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space


class OffPolicyBuffer:
    """Off-policy buffer."""

    def __init__(self, args, share_obs_space, num_agents, obs_spaces, act_spaces):
        """Initialize off-policy buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
            num_agents: (int) number of agents
            obs_spaces: (gym.Space or list) observation spaces
            act_spaces: (gym.Space) action spaces
        """
        self.buffer_size = args["buffer_size"]
        self.batch_size = args["batch_size"]
        self.n_step = args["n_step"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.gamma = args["gamma"]
        self.cur_size = 0  # current occupied size of buffer
        self.idx = 0  # current index to insert
        self.num_agents = num_agents

        # get shapes of share obs, obs, and actions
        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        if isinstance(share_obs_shape[-1], list):
            share_obs_shape = share_obs_shape[:1]
        obs_shapes = []
        act_shapes = []
        for agent_id in range(num_agents):
            obs_shape = get_shape_from_obs_space(obs_spaces[agent_id])
            if isinstance(obs_shape[-1], list):
                obs_shape = obs_shape[:1]
            obs_shapes.append(obs_shape)
            act_shapes.append(get_shape_from_act_space(act_spaces[agent_id]))

        # Buffer for share observations
        self.share_obs = np.zeros((self.buffer_size, *share_obs_shape), dtype=np.float32)

        # Buffer for next share observations
        self.next_share_obs = np.zeros((self.buffer_size, *share_obs_shape), dtype=np.float32)

        # Buffer for observations and next observations of each agent
        self.obs = []
        self.next_obs = []
        for agent_id in range(num_agents):
            self.obs.append(np.zeros((self.buffer_size, *obs_shapes[agent_id]), dtype=np.float32))
            self.next_obs.append(
                np.zeros((self.buffer_size, *obs_shapes[agent_id]), dtype=np.float32)
            )

        # Buffer for rewards received by agents at each timestep
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)

        # Buffer for actions taken by agents at each timestep
        self.actions = []
        for agent_id in range(num_agents):
            self.actions.append(
                np.zeros((self.buffer_size, act_shapes[agent_id]), dtype=np.float32)
            )

        # Buffer for done and termination flags
        self.dones = np.full((self.buffer_size, 1), False)
        self.terms = np.full((self.buffer_size, 1), False)

    def insert(self, data):
        """Insert data into buffer.
        Args:
            data: a tuple of (share_obs, obs, action, reward, done, term, next_share_obs, next_obs)
            share_obs: (n_rollout_threads, *share_obs_shape)
            obs: [(n_rollout_threads, *obs_shapes[agent_id]) for agent_id in range(num_agents)]
            action: [(n_rollout_threads, *act_shapes[agent_id]) for agent_id in range(num_agents)]
            reward: (n_rollout_threads, 1)
            done: (n_rollout_threads, 1)
            term: (n_rollout_threads, 1)
            next_share_obs: (n_rollout_threads, *share_obs_shape)
            next_obs: [(n_rollout_threads, *obs_shapes[agent_id]) for agent_id in range(num_agents)]
        """
        share_obs, obs, actions, reward, done, term, next_share_obs, next_obs = data
        length = share_obs.shape[0]
        if self.idx + length <= self.buffer_size:  # no overflow
            s = self.idx
            e = self.idx + length
            self.share_obs[s:e] = share_obs.copy()
            self.rewards[s:e] = reward.copy()
            self.dones[s:e] = done.copy()
            self.terms[s:e] = term.copy()
            self.next_share_obs[s:e] = next_share_obs.copy()
            for agent_id in range(self.num_agents):
                self.obs[agent_id][s:e] = obs[agent_id].copy()
                self.actions[agent_id][s:e] = actions[agent_id].copy()
                self.next_obs[agent_id][s:e] = next_obs[agent_id].copy()
        else:  # overflow
            len1 = self.buffer_size - self.idx  # length of first segment
            len2 = length - len1  # length of second segment

            # insert first segment
            s = self.idx
            e = self.buffer_size
            self.share_obs[s:e] = share_obs[0:len1].copy()
            self.rewards[s:e] = reward[0:len1].copy()
            self.dones[s:e] = done[0:len1].copy()
            self.terms[s:e] = term[0:len1].copy()
            self.next_share_obs[s:e] = next_share_obs[0:len1].copy()
            for agent_id in range(self.num_agents):
                self.obs[agent_id][s:e] = obs[agent_id][0:len1].copy()
                self.actions[agent_id][s:e] = actions[agent_id][0:len1].copy()
                self.next_obs[agent_id][s:e] = next_obs[agent_id][0:len1].copy()

            # insert second segment
            s = 0
            e = len2
            self.share_obs[s:e] = share_obs[len1:length].copy()
            self.rewards[s:e] = reward[len1:length].copy()
            self.dones[s:e] = done[len1:length].copy()
            self.terms[s:e] = term[len1:length].copy()
            self.next_share_obs[s:e] = next_share_obs[len1:length].copy()
            for agent_id in range(self.num_agents):
                self.obs[agent_id][s:e] = obs[agent_id][len1:length].copy()
                self.actions[agent_id][s:e] = actions[agent_id][len1:length].copy()
                self.next_obs[agent_id][s:e] = next_obs[agent_id][len1:length].copy()
        
        self.idx = (self.idx + length) % self.buffer_size  # update index
        self.cur_size = min(self.cur_size + length, self.buffer_size)  # update current size

    def sample(self):
        """Sample data for training.
        Returns:
            sp_share_obs: (batch_size, *dim)
            sp_obs: (n_agents, batch_size, *dim)
            sp_actions: (n_agents, batch_size, *dim)
            sp_reward: (batch_size, 1)
            sp_done: (batch_size, 1)
            sp_term: (batch_size, 1)
            sp_next_share_obs: (batch_size, *dim)
            sp_next_obs: (n_agents, batch_size, *dim)
        """
        self.update_end_flag()  # update the current end flag
        indice = torch.randperm(self.cur_size).numpy()[: self.batch_size]  # sample indice

        # get data at the beginning indice
        sp_share_obs = self.share_obs[indice]  
        sp_obs = np.array([self.obs[agent_id][indice] for agent_id in range(self.num_agents)])
        sp_actions = np.array(
            [self.actions[agent_id][indice] for agent_id in range(self.num_agents)]
        )

        # compute the indices along n steps
        indices = [indice]
        for _ in range(self.n_step - 1):
            indices.append(self.next(indices[-1]))

        # get data at the last indice
        sp_done = self.dones[indices[-1]]
        sp_term = self.terms[indices[-1]]
        sp_next_share_obs = self.next_share_obs[indices[-1]]
        sp_next_obs = np.array(
            [self.next_obs[agent_id][indices[-1]] for agent_id in range(self.num_agents)]
        )

        # compute accumulated rewards and the corresponding gamma
        gamma_buffer = np.ones(self.n_step + 1)
        for i in range(1, self.n_step + 1):
            gamma_buffer[i] = gamma_buffer[i - 1] * self.gamma
        sp_reward = np.zeros((self.batch_size, 1))
        gammas = np.full(self.batch_size, self.n_step)
        for n in range(self.n_step - 1, -1, -1):
            now = indices[n]
            gammas[self.end_flag[now] > 0] = n + 1
            sp_reward[self.end_flag[now] > 0] = 0.0
            sp_reward = self.rewards[now] + self.gamma * sp_reward
        sp_gamma = gamma_buffer[gammas].reshape(self.batch_size, 1)
        
        return (
            sp_share_obs,
            sp_obs,
            sp_actions,
            sp_reward,
            sp_done,
            sp_term,
            sp_next_share_obs,
            sp_next_obs,
            sp_gamma,
        )

    def next(self, indices):
        """Get next indices"""
        return (indices + (1 - self.end_flag[indices]) * self.n_rollout_threads) % self.buffer_size

    def update_end_flag(self):
        """Update current end flag for computing n-step return.
        End flag is True at the steps which are the end of an episode or the latest but unfinished steps.
        """
        self.unfinished_index = (
            self.idx - np.arange(self.n_rollout_threads) - 1 + self.cur_size
        ) % self.cur_size
        self.end_flag = self.dones.copy().squeeze()
        self.end_flag[self.unfinished_index] = True

    def get_mean_rewards(self):
        """Get mean rewards of the buffer"""
        return np.mean(self.rewards[: self.cur_size])
