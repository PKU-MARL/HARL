"""HADDPG algorithm."""
from copy import deepcopy
import torch
from harl.models.policy_models.deterministic_policy import DeterministicPolicy
from harl.utils.envs_tools import check
from harl.algorithms.actors.off_policy_base import OffPolicyBase


class HADDPG(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        assert (
            act_space.__class__.__name__ == "Box"
        ), f"only continuous action space is supported by {self.__class__.__name__}."
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.expl_noise = args["expl_noise"]

        self.actor = DeterministicPolicy(args, obs_space, act_space, device)
        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.low = torch.tensor(act_space.low).to(**self.tpdv)
        self.high = torch.tensor(act_space.high).to(**self.tpdv)
        self.scale = (self.high - self.low) / 2
        self.mean = (self.high + self.low) / 2
        self.turn_off_grad()

    def get_actions(self, obs, add_noise):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            add_noise: (bool) whether to add noise
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.actor(obs)
        if add_noise:
            actions += torch.randn_like(actions) * self.expl_noise * self.scale
            actions = torch.max(torch.min(actions, self.high), self.low)
        return actions

    def get_target_actions(self, obs):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        return self.target_actor(obs)
