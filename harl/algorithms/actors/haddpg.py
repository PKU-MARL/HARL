"""HADDPG algorithm."""
from copy import deepcopy
import torch
from harl.models.policy_models.deterministic_policy import DeterministicPolicy
from harl.models.policy_models.stochastic_mlp_policy import StochasticMlpPolicy
from harl.utils.envs_tools import check
from harl.utils.discrete_util import gumbel_softmax
from harl.algorithms.actors.off_policy_base import OffPolicyBase


class HADDPG(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.expl_noise = args["expl_noise"]
        self.device = device
        self.action_type = act_space.__class__.__name__

        if act_space.__class__.__name__ == "Box":
            self.actor = DeterministicPolicy(args, obs_space, act_space, device)
        else:
            self.actor = StochasticMlpPolicy(args, obs_space, act_space, device)
        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        if act_space.__class__.__name__ == "Box":
            self.low = torch.tensor(act_space.low).to(**self.tpdv)
            self.high = torch.tensor(act_space.high).to(**self.tpdv)
            self.scale = (self.high - self.low) / 2
            self.mean = (self.high + self.low) / 2
        self.turn_off_grad()

    def get_actions(self, obs, available_actions=None, add_noise=False, onehot=False):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            add_noise: (bool) whether to add noise
            onehot: (bool) whether to use onehot action
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        if self.action_type == "Box":
            actions = self.actor(obs)
            if add_noise:
                actions += torch.randn_like(actions) * self.expl_noise * self.scale
                actions = torch.max(torch.min(actions, self.high), self.low)
        elif self.action_type == "Discrete":  # TODO: exploration
            if onehot:
                logits = self.actor.get_logits(obs, available_actions)
                actions = gumbel_softmax(
                    logits, hard=True, device=self.device
                )  # onehot actions
            else:
                actions = self.actor(obs, available_actions, stochastic=add_noise)
        elif self.action_type == "MultiDiscrete":
            if onehot:
                logits = self.actor.get_logits(obs, available_actions)
                actions = []
                for logit in logits:
                    action = gumbel_softmax(
                        logit, hard=True, device=self.device
                    )  # onehot actions
                    actions.append(action)
                actions = torch.cat(actions, dim=-1)
            else:
                actions = self.actor(obs, available_actions, stochastic=add_noise)
        return actions

    def get_target_actions(self, obs, available_actions=None):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        if self.action_type == "Box":
            actions = self.target_actor(obs)
        elif self.action_type == "Discrete":
            logits = self.target_actor.get_logits(obs, available_actions)
            actions = gumbel_softmax(
                logits, hard=True, device=self.device
            )  # onehot actions for critic training
        elif self.action_type == "MultiDiscrete":
            logits = self.target_actor.get_logits(obs, available_actions)
            actions = []
            for logit in logits:
                action = gumbel_softmax(
                    logit, hard=True, device=self.device
                )  # onehot actions for critic training
                actions.append(action)
            actions = torch.cat(actions, dim=-1)
        return actions
