"""Base class for off-policy algorithms."""

from copy import deepcopy
import numpy as np
import torch
from harl.utils.envs_tools import check
from harl.utils.models_tools import update_linear_schedule


class OffPolicyBase:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        pass

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        update_linear_schedule(self.actor_optimizer, step, steps, self.lr)

    def get_actions(self, obs, randomness):
        pass

    def get_target_actions(self, obs):
        pass

    def soft_update(self):
        """Soft update target actor."""
        for param_target, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def save(self, save_dir, id):
        """Save the actor and target actor."""
        torch.save(
            self.actor.state_dict(), str(save_dir) + "/actor_agent" + str(id) + ".pt"
        )
        torch.save(
            self.target_actor.state_dict(),
            str(save_dir) + "/target_actor_agent" + str(id) + ".pt",
        )

    def restore(self, model_dir, id):
        """Restore the actor and target actor."""
        actor_state_dict = torch.load(str(model_dir) + "/actor_agent" + str(id) + ".pt")
        self.actor.load_state_dict(actor_state_dict)
        target_actor_state_dict = torch.load(
            str(model_dir) + "/target_actor_agent" + str(id) + ".pt"
        )
        self.target_actor.load_state_dict(target_actor_state_dict)

    def turn_on_grad(self):
        """Turn on grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = True

    def turn_off_grad(self):
        """Turn off grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = False
