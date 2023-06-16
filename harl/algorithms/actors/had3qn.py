"""HAD3QN algorithm."""
from copy import deepcopy
import numpy as np
import torch
from harl.models.value_function_models.dueling_q_net import DuelingQNet
from harl.utils.envs_tools import check
from harl.algorithms.actors.off_policy_base import OffPolicyBase


class HAD3QN(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        assert (
            act_space.__class__.__name__ == "Discrete"
        ), "only discrete action space is supported by HAD3QN."
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.tpdv_a = dict(dtype=torch.int64, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.epsilon = args["epsilon"]
        self.action_dim = act_space.n

        self.actor = DuelingQNet(args, obs_space, self.action_dim, device)
        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.turn_off_grad()

    def get_actions(self, obs, epsilon_greedy):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            epsilon_greedy: (bool) whether choose action epsilon-greedily
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, 1) or (batch_size, 1)
        """
        obs = check(obs).to(**self.tpdv)
        if np.random.random() < self.epsilon and epsilon_greedy:
            actions = torch.randint(
                low=0, high=self.action_dim, size=(*obs.shape[:-1], 1)
            )
        else:
            actions = self.actor(obs).argmax(dim=-1, keepdim=True)
        return actions

    def get_target_actions(self, obs):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, 1)
        """
        obs = check(obs).to(**self.tpdv)
        return self.target_actor(obs).argmax(dim=-1, keepdim=True)

    def train_values(self, obs, actions):
        """Get values with grad for obs and actions
        Args:
            obs: (np.ndarray) observations batch, shape is (batch_size, dim)
            actions: (torch.Tensor) actions batch, shape is (batch_size, 1)
        Returns:
            values: (torch.Tensor) values predicted by Q network, shape is (batch_size, 1)
        """
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv_a)
        values = torch.gather(input=self.actor(obs), dim=1, index=actions)
        return values
