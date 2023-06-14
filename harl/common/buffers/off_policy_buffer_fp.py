"""Off-policy buffer."""
import numpy as np
import torch
from harl.common.buffers.off_policy_buffer_base import OffPolicyBufferBase


class OffPolicyBufferFP(OffPolicyBufferBase):
    """Off-policy buffer that uses Feature-Pruned (FP) state.
    When FP state is used, the critic takes different global state as input for different actors. Thus, OffPolicyBufferFP has an extra dimension for number of agents.
    """

    def __init__(self, args, share_obs_space, num_agents, obs_spaces, act_spaces):
        """Initialize off-policy buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
            num_agents: (int) number of agents
            obs_spaces: (gym.Space or list) observation spaces
            act_spaces: (gym.Space) action spaces
        """
        super(OffPolicyBufferFP, self).__init__(args, share_obs_space, num_agents, obs_spaces, act_spaces)

        # Buffer for share observations
        self.share_obs = np.zeros((self.buffer_size, self.num_agents, *self.share_obs_shape), dtype=np.float32)

        # Buffer for next share observations
        self.next_share_obs = np.zeros((self.buffer_size, self.num_agents, *self.share_obs_shape), dtype=np.float32)

        # Buffer for rewards received by agents at each timestep
        self.rewards = np.zeros((self.buffer_size, self.num_agents, 1), dtype=np.float32)

        # Buffer for done and termination flags
        self.dones = np.full((self.buffer_size, self.num_agents, 1), False)
        self.terms = np.full((self.buffer_size, self.num_agents, 1), False)

    def sample(self):
        """Sample data for training.
        Returns:
            sp_share_obs: (n_agents * batch_size, *dim)
            sp_obs: (n_agents, batch_size, *dim)
            sp_actions: (n_agents, batch_size, *dim)
            sp_available_actions: (n_agents, batch_size, *dim)
            sp_reward: (n_agents * batch_size, 1)
            sp_done: (n_agents * batch_size, 1)
            sp_valid_transitions: (n_agents, batch_size, 1)
            sp_term: (n_agents * batch_size, 1)
            sp_next_share_obs: (n_agents * batch_size, *dim)
            sp_next_obs: (n_agents, batch_size, *dim)
            sp_next_available_actions: (n_agents, batch_size, *dim)
            sp_gamma: (n_agents * batch_size, 1)
        """
        self.update_end_flag()  # update the current end flag
        indice = torch.randperm(self.cur_size).numpy()[: self.batch_size]  # sample indice, shape: (batch_size, )

        # get data at the beginning indice
        sp_share_obs = np.concatenate(
            self.share_obs[indice].transpose(1, 0, 2), axis=0
        )  # (batch_size, n_agents, *dim) -> (n_agents, batch_size, *dim) -> (n_agents * batch_size, *dim)
        sp_obs = np.array([self.obs[agent_id][indice] for agent_id in range(self.num_agents)])
        sp_actions = np.array(
            [self.actions[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        sp_valid_transitions = np.array(
            [self.valid_transitions[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        if self.act_spaces[0].__class__.__name__ == 'Discrete':
            sp_available_actions = np.array(
                [self.available_actions[agent_id][indice] for agent_id in range(self.num_agents)]
            )

        # compute the indices along n steps
        indice = np.repeat(np.expand_dims(indice, axis=-1), self.num_agents, axis=-1)  # (batch_size, n_agents)
        indices = [indice]
        for _ in range(self.n_step - 1):
            indices.append(self.next(indices[-1]))

        # get data at the last indice
        sp_done = np.concatenate(
            [self.dones[indices[-1][:, agent_id], agent_id] for agent_id in range(self.num_agents)]
        )  # (n_agents, batch_size, 1) -> (n_agents * batch_size, 1)
        sp_term = np.concatenate(
            [self.terms[indices[-1][:, agent_id], agent_id] for agent_id in range(self.num_agents)]
        )  # (n_agents, batch_size, 1) -> (n_agents * batch_size, 1)
        sp_next_share_obs = np.concatenate(
            [self.next_share_obs[indices[-1][:, agent_id], agent_id] for agent_id in range(self.num_agents)]
        )  # (n_agents, batch_size, *dim) -> (n_agents * batch_size, *dim)
        sp_next_obs = np.array(
            [self.next_obs[agent_id][indices[-1][:, agent_id]] for agent_id in range(self.num_agents)]
        )
        if self.act_spaces[0].__class__.__name__ == 'Discrete':
            sp_next_available_actions = np.array(
                [self.next_available_actions[agent_id][indices[-1][:, agent_id]] for agent_id in range(self.num_agents)]
            )

        # compute accumulated rewards and the corresponding gamma
        gamma_buffer = np.ones((self.num_agents, self.n_step + 1))
        for i in range(1, self.n_step + 1):
            gamma_buffer[:, i] = gamma_buffer[:, i - 1] * self.gamma
        sp_reward = np.zeros((self.batch_size, self.num_agents, 1))
        gammas = np.full((self.batch_size, self.num_agents), self.n_step)
        for n in range(self.n_step - 1, -1, -1):
            now = indices[n]
            end_flag = np.column_stack(
                [self.end_flag[now[:, agent_id], agent_id] for agent_id in range(self.num_agents)]
            )
            gammas[end_flag > 0] = n + 1
            sp_reward[end_flag > 0] = 0.0
            rewards = np.expand_dims(
                np.column_stack([self.rewards[now[:, agent_id], agent_id] for agent_id in range(self.num_agents)]), axis=-1
            )
            sp_reward = rewards + self.gamma * sp_reward
        sp_reward = np.concatenate(
            sp_reward.transpose(1, 0, 2), axis=0
        )
        sp_gamma = np.concatenate(
           [gamma_buffer[agent_id][gammas[:, agent_id]] for agent_id in range(self.num_agents)]
        ).reshape(-1, 1)  # (n_agents * batch_size, ) -> (n_agents * batch_size, 1)

        if self.act_spaces[0].__class__.__name__ == 'Discrete':
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
                sp_gamma
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
        end_flag = np.column_stack(
            [self.end_flag[indices[:, agent_id], agent_id] for agent_id in range(self.num_agents)]
        )  # (batch_size, n_agents)
        return (indices + (1 - end_flag) * self.n_rollout_threads) % self.buffer_size

    def update_end_flag(self):
        """Update current end flag for computing n-step return.
        End flag is True at the steps which are the end of an episode or the latest but unfinished steps.
        """
        self.unfinished_index = (
            self.idx - np.arange(self.n_rollout_threads) - 1 + self.cur_size
        ) % self.cur_size
        self.end_flag = self.dones.copy().squeeze()  # FP: (batch_size, n_agents)
        self.end_flag[self.unfinished_index, :] = True

