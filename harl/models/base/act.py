import torch
import torch.nn as nn
from harl.models.base.distributions import Categorical, DiagGaussian


class ACTLayer(nn.Module):
    """MLP Module to compute actions."""

    def __init__(
        self, action_space, inputs_dim, initialization_method, gain, args=None
    ):
        """Initialize ACTLayer.
        Args:
            action_space: (gym.Space) action space.
            inputs_dim: (int) dimension of network input.
            initialization_method: (str) initialization method.
            gain: (float) gain of the output layer of the network.
            args: (dict) arguments relevant to the network.
        """
        super(ACTLayer, self).__init__()
        self.action_type = action_space.__class__.__name__
        self.multidiscrete_action = False

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(
                inputs_dim, action_dim, initialization_method, gain
            )
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(
                inputs_dim, action_dim, initialization_method, gain, args
            )
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multidiscrete_action = True
            action_dims = action_space.nvec
            action_outs = []
            for action_dim in action_dims:
                action_outs.append(
                    Categorical(inputs_dim, action_dim, initialization_method, gain)
                )
            self.action_outs = nn.ModuleList(action_outs)

    def forward(self, x, available_actions=None, deterministic=False):
        """Compute actions and action logprobs from given input.
        Args:
            x: (torch.Tensor) input to network.
            available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """

        if self.multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_distribution = action_out(x, available_actions)
                action = (
                    action_distribution.mode()
                    if deterministic
                    else action_distribution.sample()
                )
                action_log_prob = action_distribution.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(
                dim=-1, keepdim=True
            )
        else:
            action_distribution = self.action_out(x, available_actions)
            actions = (
                action_distribution.mode()
                if deterministic
                else action_distribution.sample()
            )
            action_log_probs = action_distribution.log_probs(actions)

        return actions, action_log_probs

    def get_logits(self, x, available_actions=None):
        """Get action logits from inputs.
        Args:
            x: (torch.Tensor) input to network.
            available_actions: (torch.Tensor) denotes which actions are available to agent
                                      (if None, all actions available)
        Returns:
            action_logits: (torch.Tensor) logits of actions for the given inputs.
        """
        if self.multidiscrete_action:
            action_logits = []
            for action_out in self.action_outs:
                action_distribution = action_out(x, available_actions)
                action_logits.append(action_distribution.logits)
        else:
            action_distribution = self.action_out(x, available_actions)
            action_logits = action_distribution.logits

        return action_logits

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            x: (torch.Tensor) input to network.
            action: (torch.Tensor) actions whose entropy and log probability to evaluate.
            available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        if self.multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_distribution = action_out(x)
                action_log_probs.append(
                    action_distribution.log_probs(act.unsqueeze(-1))
                )
                if active_masks is not None:
                    dist_entropy.append(
                        (action_distribution.entropy() * active_masks)
                        / active_masks.sum()
                    )
                else:
                    dist_entropy.append(
                        action_distribution.entropy() / action_log_probs[-1].size(0)
                    )
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(
                dim=-1, keepdim=True
            )
            dist_entropy = (
                torch.cat(dist_entropy, dim=-1).sum(dim=-1, keepdim=True).mean()
            )
            return action_log_probs, dist_entropy, None
        else:
            action_distribution = self.action_out(x, available_actions)
            action_log_probs = action_distribution.log_probs(action)
            if active_masks is not None:
                if self.action_type == "Discrete":
                    dist_entropy = (
                        action_distribution.entropy() * active_masks.squeeze(-1)
                    ).sum() / active_masks.sum()
                else:
                    dist_entropy = (
                        action_distribution.entropy() * active_masks
                    ).sum() / active_masks.sum()
            else:
                dist_entropy = action_distribution.entropy().mean()

        return action_log_probs, dist_entropy, action_distribution
