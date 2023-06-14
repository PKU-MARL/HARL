"""HATD3 algorithm."""
import torch
from harl.utils.envs_tools import check
from harl.algorithms.actors.haddpg import HADDPG


class HATD3(HADDPG):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, act_space, device)
        self.policy_noise = args["policy_noise"]
        self.noise_clip = args["noise_clip"]

    def get_target_actions(self, obs):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.target_actor(obs)
        noise = torch.randn_like(actions) * self.policy_noise * self.scale
        noise = torch.max(
            torch.min(noise, self.noise_clip * self.scale),
            -self.noise_clip * self.scale,
        )
        actions += noise
        actions = torch.max(torch.min(actions, self.high), self.low)
        return actions
