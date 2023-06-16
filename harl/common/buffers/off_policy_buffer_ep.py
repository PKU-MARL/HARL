"""Off-policy buffer."""
import numpy as np
import torch
from harl.common.buffers.off_policy_buffer_base import OffPolicyBufferBase


class OffPolicyBufferEP(OffPolicyBufferBase):
    """Off-policy buffer that uses Environment-Provided (EP) state."""

    def __init__(self, args, share_obs_space, num_agents, obs_spaces, act_spaces):
        """Initialize off-policy buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
            num_agents: (int) number of agents
            obs_spaces: (gym.Space or list) observation spaces
            act_spaces: (gym.Space) action spaces
        """
        super(OffPolicyBufferEP, self).__init__(
            args, share_obs_space, num_agents, obs_spaces, act_spaces
        )

        # Buffer for share observations
        self.share_obs = np.zeros(
            (self.buffer_size, *self.share_obs_shape), dtype=np.float32
        )

        # Buffer for next share observations
        self.next_share_obs = np.zeros(
            (self.buffer_size, *self.share_obs_shape), dtype=np.float32
        )

        # Buffer for rewards received by agents at each timestep
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)

        # Buffer for done and termination flags
        self.dones = np.full((self.buffer_size, 1), False)
        self.terms = np.full((self.buffer_size, 1), False)

    def sample(self):
        """Sample data for training.
        Returns:
            sp_share_obs: (batch_size, *dim)
            sp_obs: (n_agents, batch_size, *dim)
            sp_actions: (n_agents, batch_size, *dim)
            sp_available_actions: (n_agents, batch_size, *dim)
            sp_reward: (batch_size, 1)
            sp_done: (batch_size, 1)
            sp_valid_transitions: (n_agents, batch_size, 1)
            sp_term: (batch_size, 1)
            sp_next_share_obs: (batch_size, *dim)
            sp_next_obs: (n_agents, batch_size, *dim)
            sp_next_available_actions: (n_agents, batch_size, *dim)
            sp_gamma: (batch_size, 1)
        """
        self.update_end_flag()  # update the current end flag
        indice = torch.randperm(self.cur_size).numpy()[
            : self.batch_size
        ]  # sample indice, shape: (batch_size, )

        # get data at the beginning indice
        sp_share_obs = self.share_obs[indice]
        sp_obs = np.array(
            [self.obs[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        sp_actions = np.array(
            [self.actions[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        sp_valid_transitions = np.array(
            [
                self.valid_transitions[agent_id][indice]
                for agent_id in range(self.num_agents)
            ]
        )
        if self.act_spaces[0].__class__.__name__ == "Discrete":
            sp_available_actions = np.array(
                [
                    self.available_actions[agent_id][indice]
                    for agent_id in range(self.num_agents)
                ]
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
            [
                self.next_obs[agent_id][indices[-1]]
                for agent_id in range(self.num_agents)
            ]
        )
        if self.act_spaces[0].__class__.__name__ == "Discrete":
            sp_next_available_actions = np.array(
                [
                    self.next_available_actions[agent_id][indices[-1]]
                    for agent_id in range(self.num_agents)
                ]
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

        if self.act_spaces[0].__class__.__name__ == "Discrete":
            return (
                sp_share_obs,
                sp_obs,
                sp_actions,
                sp_available_actions,
                sp_reward,
                sp_done,
                sp_valid_transitions,
                sp_term,
                sp_next_share_obs,
                sp_next_obs,
                sp_next_available_actions,
                sp_gamma,
            )
        else:
            return (
                sp_share_obs,
                sp_obs,
                sp_actions,
                None,
                sp_reward,
                sp_done,
                sp_valid_transitions,
                sp_term,
                sp_next_share_obs,
                sp_next_obs,
                None,
                sp_gamma,
            )

    def next(self, indices):
        """Get next indices"""
        return (
            indices + (1 - self.end_flag[indices]) * self.n_rollout_threads
        ) % self.buffer_size

    def update_end_flag(self):
        """Update current end flag for computing n-step return.
        End flag is True at the steps which are the end of an episode or the latest but unfinished steps.
        """
        self.unfinished_index = (
            self.idx - np.arange(self.n_rollout_threads) - 1 + self.cur_size
        ) % self.cur_size
        self.end_flag = self.dones.copy().squeeze()  # (batch_size, )
        self.end_flag[self.unfinished_index] = True
