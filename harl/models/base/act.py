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
        action_distribution = self.action_out(x, available_actions)
        actions = (
            action_distribution.mode()
            if deterministic
            else action_distribution.sample()
        )
        action_log_probs = action_distribution.log_probs(actions)

        return actions, action_log_probs

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
