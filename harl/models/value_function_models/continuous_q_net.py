import torch
import torch.nn as nn
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP
from harl.utils.envs_tools import get_shape_from_obs_space


def get_combined_dim(cent_obs_feature_dim, act_spaces):
    """Get the combined dimension of central observation and individual actions."""
    combined_dim = cent_obs_feature_dim
    for space in act_spaces:
        if space.__class__.__name__ == "Box":
            combined_dim += space.shape[0]
        elif space.__class__.__name__ == "Discrete":
            combined_dim += space.n
        else:
            action_dims = space.nvec
            for action_dim in action_dims:
                combined_dim += action_dim
    return combined_dim


class ContinuousQNet(nn.Module):
    """Q Network for continuous and discrete action space. Outputs the q value given global states and actions.
    Note that the name ContinuousQNet emphasizes its structure that takes observations and actions as input and outputs
    the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space.
    """

    def __init__(self, args, cent_obs_space, act_spaces, device=torch.device("cpu")):
        super(ContinuousQNet, self).__init__()
        activation_func = args["activation_func"]
        hidden_sizes = args["hidden_sizes"]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if len(cent_obs_shape) == 3:
            self.feature_extractor = PlainCNN(
                cent_obs_shape, hidden_sizes[0], activation_func
            )
            cent_obs_feature_dim = hidden_sizes[0]
        else:
            self.feature_extractor = None
            cent_obs_feature_dim = cent_obs_shape[0]
        sizes = (
            [get_combined_dim(cent_obs_feature_dim, act_spaces)]
            + list(hidden_sizes)
            + [1]
        )
        self.mlp = PlainMLP(sizes, activation_func)
        self.to(device)

    def forward(self, cent_obs, actions):
        if self.feature_extractor is not None:
            feature = self.feature_extractor(cent_obs)
        else:
            feature = cent_obs
        concat_x = torch.cat([feature, actions], dim=-1)
        q_values = self.mlp(concat_x)
        return q_values
