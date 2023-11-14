"""Modify standard PyTorch distributions so they to make compatible with this codebase."""
import torch
import torch.nn as nn
from harl.utils.models_tools import init, get_init_method


class FixedCategorical(torch.distributions.Categorical):
    """Modify standard PyTorch Categorical."""

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):
    """Modify standard PyTorch Normal."""

    def log_probs(self, actions):
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class Categorical(nn.Module):
    """A linear layer followed by a Categorical distribution."""

    def __init__(
        self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01
    ):
        super(Categorical, self).__init__()
        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    """A linear layer followed by a Diagonal Gaussian distribution."""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        initialization_method="orthogonal_",
        gain=0.01,
        args=None,
    ):
        super(DiagGaussian, self).__init__()

        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if args is not None:
            self.std_x_coef = args["std_x_coef"]
            self.std_y_coef = args["std_y_coef"]
        else:
            self.std_x_coef = 1.0
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)
