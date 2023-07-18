"""HATD3 algorithm."""
import torch
from harl.utils.envs_tools import check
from harl.utils.discrete_util import gumbel_softmax
from harl.algorithms.actors.haddpg import HADDPG


class HATD3(HADDPG):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, act_space, device)
        self.policy_noise = args["policy_noise"]
        self.noise_clip = args["noise_clip"]

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
            noise = torch.randn_like(actions) * self.policy_noise * self.scale
            noise = torch.max(
                torch.min(noise, self.noise_clip * self.scale),
                -self.noise_clip * self.scale,
            )
            actions += noise
            actions = torch.max(torch.min(actions, self.high), self.low)
        elif self.action_type == "Discrete":  # TODO: exploration
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
