"""Continuous Q Critic."""
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from harl.models.value_function_models.continuous_q_net import ContinuousQNet
from harl.utils.envs_tools import check
from harl.utils.models_tools import update_linear_schedule


class ContinuousQCritic:
    """Continuous Q Critic.
    Critic that learns a Q-function. The action space is continuous.
    Note that the name ContinuousQCritic emphasizes its structure that takes observations and actions as input and
    outputs the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space. For now, it only supports continuous action space, but we will enhance its capability to
    include discrete action space in the future.
    """

    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
    ):
        """Initialize the critic."""
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.tpdv_a = dict(dtype=torch.int64, device=device)
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.action_type = act_space[0].__class__.__name__
        self.critic = ContinuousQNet(args, share_obs_space, act_space, device)
        self.target_critic = deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
        self.use_huber_loss = args["use_huber_loss"]
        self.huber_delta = args["huber_delta"]
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )
        self.turn_off_grad()

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        update_linear_schedule(self.critic_optimizer, step, steps, self.critic_lr)

    def soft_update(self):
        """Soft update the target network."""
        for param_target, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def get_values(self, share_obs, actions):
        """Get the Q values."""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        return self.critic(share_obs, actions)

    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        valid_transition,
        term,
        next_share_obs,
        next_actions,
        gamma,
        value_normalizer=None,
    ):
        """Train the critic.
        Args:
            share_obs: (np.ndarray) shape is (batch_size, dim)
            actions: (np.ndarray) shape is (n_agents, batch_size, dim)
            reward: (np.ndarray) shape is (batch_size, 1)
            done: (np.ndarray) shape is (batch_size, 1)
            valid_transition: (np.ndarray) shape is (n_agents, batch_size, 1)
            term: (np.ndarray) shape is (batch_size, 1)
            next_share_obs: (np.ndarray) shape is (batch_size, dim)
            next_actions: (np.ndarray) shape is (n_agents, batch_size, dim)
            gamma: (np.ndarray) shape is (batch_size, 1)
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        """
        assert share_obs.__class__.__name__ == "ndarray"
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert valid_transition.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert next_share_obs.__class__.__name__ == "ndarray"
        assert gamma.__class__.__name__ == "ndarray"

        share_obs = check(share_obs).to(**self.tpdv)
        if self.action_type == "Box":
            actions = check(actions).to(**self.tpdv)
            actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)
        else:
            actions = check(actions).to(**self.tpdv_a)
            one_hot_actions = []
            for agent_id in range(len(actions)):
                if self.action_type == "MultiDiscrete":
                    action_dims = self.act_space[agent_id].nvec
                    one_hot_action = []
                    for dim in range(len(action_dims)):
                        one_hot = F.one_hot(
                            actions[agent_id, :, dim], num_classes=action_dims[dim]
                        )
                        one_hot_action.append(one_hot)
                    one_hot_action = torch.cat(one_hot_action, dim=-1)
                else:
                    one_hot_action = F.one_hot(
                        actions[agent_id], num_classes=self.act_space[agent_id].n
                    )
                one_hot_actions.append(one_hot_action)
            actions = torch.squeeze(torch.cat(one_hot_actions, dim=-1), dim=1).to(
                **self.tpdv_a
            )
        if self.state_type == "FP":
            actions = torch.tile(actions, (self.num_agents, 1))
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        valid_transition = check(np.concatenate(valid_transition, axis=0)).to(
            **self.tpdv
        )
        term = check(term).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        if self.action_type == "Box":
            next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
        else:
            next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv_a)
        if self.state_type == "FP":
            next_actions = torch.tile(next_actions, (self.num_agents, 1))
        gamma = check(gamma).to(**self.tpdv)
        next_q_values = self.target_critic(next_share_obs, next_actions)
        if self.use_proper_time_limits:
            if value_normalizer is not None:
                q_targets = reward + gamma * check(
                    value_normalizer.denormalize(next_q_values)
                ).to(**self.tpdv) * (1 - term)
                value_normalizer.update(q_targets)
                q_targets = check(value_normalizer.normalize(q_targets)).to(**self.tpdv)
            else:
                q_targets = reward + gamma * next_q_values * (1 - term)
        else:
            if value_normalizer is not None:
                q_targets = reward + gamma * check(
                    value_normalizer.denormalize(next_q_values)
                ).to(**self.tpdv) * (1 - done)
                value_normalizer.update(q_targets)
                q_targets = check(value_normalizer.normalize(q_targets)).to(**self.tpdv)
            else:
                q_targets = reward + gamma * next_q_values * (1 - done)
        if self.use_huber_loss:
            if self.state_type == "FP" and self.use_policy_active_masks:
                critic_loss = (
                    torch.sum(
                        F.huber_loss(
                            self.critic(share_obs, actions),
                            q_targets,
                            delta=self.huber_delta,
                        )
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
            else:
                critic_loss = torch.mean(
                    F.huber_loss(
                        self.critic(share_obs, actions),
                        q_targets,
                        delta=self.huber_delta,
                    )
                )
        else:
            if self.state_type == "FP" and self.use_policy_active_masks:
                critic_loss = (
                    torch.sum(
                        F.mse_loss(self.critic(share_obs, actions), q_targets)
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
            else:
                critic_loss = torch.mean(
                    F.mse_loss(self.critic(share_obs, actions), q_targets)
                )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save(self, save_dir):
        """Save the model."""
        torch.save(self.critic.state_dict(), str(save_dir) + "/critic_agent" + ".pt")
        torch.save(
            self.target_critic.state_dict(),
            str(save_dir) + "/target_critic_agent" + ".pt",
        )

    def restore(self, model_dir):
        """Restore the model."""
        critic_state_dict = torch.load(str(model_dir) + "/critic_agent" + ".pt")
        self.critic.load_state_dict(critic_state_dict)
        target_critic_state_dict = torch.load(
            str(model_dir) + "/target_critic_agent" + ".pt"
        )
        self.target_critic.load_state_dict(target_critic_state_dict)

    def turn_on_grad(self):
        """Turn on the gradient for the critic."""
        for param in self.critic.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """Turn off the gradient for the critic."""
        for param in self.critic.parameters():
            param.requires_grad = False
