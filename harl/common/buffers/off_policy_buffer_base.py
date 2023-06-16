"""Off-policy buffer."""
import numpy as np
from harl.utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space


class OffPolicyBufferBase:
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
        self.act_spaces = act_spaces

        # get shapes of share obs, obs, and actions
        self.share_obs_shape = get_shape_from_obs_space(share_obs_space)
        if isinstance(self.share_obs_shape[-1], list):
            self.share_obs_shape = self.share_obs_shape[:1]
        obs_shapes = []
        act_shapes = []
        for agent_id in range(num_agents):
            obs_shape = get_shape_from_obs_space(obs_spaces[agent_id])
            if isinstance(obs_shape[-1], list):
                obs_shape = obs_shape[:1]
            obs_shapes.append(obs_shape)
            act_shapes.append(get_shape_from_act_space(act_spaces[agent_id]))

        # Buffer for observations and next observations of each agent
        self.obs = []
        self.next_obs = []
        for agent_id in range(num_agents):
            self.obs.append(
                np.zeros((self.buffer_size, *obs_shapes[agent_id]), dtype=np.float32)
            )
            self.next_obs.append(
                np.zeros((self.buffer_size, *obs_shapes[agent_id]), dtype=np.float32)
            )

        # Buffer for valid_transitions of each agent
        self.valid_transitions = []
        for agent_id in range(num_agents):
            self.valid_transitions.append(
                np.ones((self.buffer_size, 1), dtype=np.float32)
            )

        # Buffer for actions and available actions taken by agents at each timestep
        self.actions = []
        self.available_actions = []
        self.next_available_actions = []
        for agent_id in range(num_agents):
            self.actions.append(
                np.zeros((self.buffer_size, act_shapes[agent_id]), dtype=np.float32)
            )
            if act_spaces[agent_id].__class__.__name__ == "Discrete":
                self.available_actions.append(
                    np.zeros(
                        (self.buffer_size, act_spaces[agent_id].n), dtype=np.float32
                    )
                )
                self.next_available_actions.append(
                    np.zeros(
                        (self.buffer_size, act_spaces[agent_id].n), dtype=np.float32
                    )
                )

    def insert(self, data):
        """Insert data into buffer.
        Args:
            data: a tuple of (share_obs, obs, actions, available_actions, reward, done, valid_transitions, term, next_share_obs, next_obs, next_available_actions)
            share_obs: EP: (n_rollout_threads, *share_obs_shape), FP: (n_rollout_threads, num_agents, *share_obs_shape)
            obs: [(n_rollout_threads, *obs_shapes[agent_id]) for agent_id in range(num_agents)]
            actions: [(n_rollout_threads, *act_shapes[agent_id]) for agent_id in range(num_agents)]
            available_actions: [(n_rollout_threads, *act_shapes[agent_id]) for agent_id in range(num_agents)]
            reward: EP: (n_rollout_threads, 1), FP: (n_rollout_threads, num_agents, 1)
            done: EP: (n_rollout_threads, 1), FP: (n_rollout_threads, num_agents, 1)
            valid_transitions: [(n_rollout_threads, 1) for agent_id in range(num_agents)]
            term: EP: (n_rollout_threads, 1), FP: (n_rollout_threads, num_agents, 1)
            next_share_obs: EP: (n_rollout_threads, *share_obs_shape), FP: (n_rollout_threads, num_agents, *share_obs_shape)
            next_obs: [(n_rollout_threads, *obs_shapes[agent_id]) for agent_id in range(num_agents)]
            next_available_actions: [(n_rollout_threads, *act_shapes[agent_id]) for agent_id in range(num_agents)]
        """
        (
            share_obs,
            obs,
            actions,
            available_actions,
            reward,
            done,
            valid_transitions,
            term,
            next_share_obs,
            next_obs,
            next_available_actions,
        ) = data
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
                self.valid_transitions[agent_id][s:e] = valid_transitions[
                    agent_id
                ].copy()
                if self.act_spaces[agent_id].__class__.__name__ == "Discrete":
                    self.available_actions[agent_id][s:e] = available_actions[
                        agent_id
                    ].copy()
                    self.next_available_actions[agent_id][s:e] = next_available_actions[
                        agent_id
                    ].copy()
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
                self.valid_transitions[agent_id][s:e] = valid_transitions[agent_id][
                    0:len1
                ].copy()
                if self.act_spaces[agent_id].__class__.__name__ == "Discrete":
                    self.available_actions[agent_id][s:e] = available_actions[agent_id][
                        0:len1
                    ].copy()
                    self.next_available_actions[agent_id][s:e] = next_available_actions[
                        agent_id
                    ][0:len1].copy()
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
                self.valid_transitions[agent_id][s:e] = valid_transitions[agent_id][
                    len1:length
                ].copy()
                if self.act_spaces[agent_id].__class__.__name__ == "Discrete":
                    self.available_actions[agent_id][s:e] = available_actions[agent_id][
                        len1:length
                    ].copy()
                    self.next_available_actions[agent_id][s:e] = next_available_actions[
                        agent_id
                    ][len1:length].copy()
                self.next_obs[agent_id][s:e] = next_obs[agent_id][len1:length].copy()

        self.idx = (self.idx + length) % self.buffer_size  # update index
        self.cur_size = min(
            self.cur_size + length, self.buffer_size
        )  # update current size

    def sample(self):
        pass

    def next(self, indices):
        pass

    def update_end_flag(self):
        pass

    def get_mean_rewards(self):
        """Get mean rewards of the buffer"""
        return np.mean(self.rewards[: self.cur_size])
