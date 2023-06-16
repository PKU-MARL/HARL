"""Discrete Q Critic."""
from copy import deepcopy
import torch
from harl.models.value_function_models.dueling_q_net import DuelingQNet
from harl.utils.envs_tools import check
from harl.utils.models_tools import update_linear_schedule


class DiscreteQCritic:
    """Discrete Q Critic.
    Critic that learns a Q-function. The action space is discrete.
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
        self.process_action_spaces(act_space)
        self.critic = DuelingQNet(args, share_obs_space, self.joint_action_dim, device)
        self.target_critic = deepcopy(self.critic)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
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
        """Get values for given observations and actions."""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv_a)
        joint_action = self.indiv_to_joint(actions)
        return torch.gather(self.critic(share_obs), 1, joint_action)

    def train_values(self, share_obs, actions):
        """Train the critic.
        Args:
            share_obs: shape is (batch_size, dim)
            actions: shape is (n_agents, batch_size, dim)
        """
        share_obs = check(share_obs).to(**self.tpdv)
        all_values = self.critic(share_obs)
        actions = deepcopy(actions)

        def update_actions(agent_id):
            joint_idx = self.get_joint_idx(actions, agent_id)
            values = torch.gather(all_values, 1, joint_idx)
            action = torch.argmax(values, dim=-1, keepdim=True)
            actions[agent_id] = action

        def get_values():
            joint_action = self.indiv_to_joint(actions)
            return torch.gather(all_values, 1, joint_action)

        return update_actions, get_values

    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        term,
        next_share_obs,
        next_actions,
        gamma,
    ):
        """Update the critic.
        Args:
            share_obs: (np.ndarray) shape is (batch_size, dim)
            actions: (np.ndarray) shape is (n_agents, batch_size, dim)
            reward: (np.ndarray) shape is (batch_size, 1)
            done: (np.ndarray) shape is (batch_size, 1)
            term: (np.ndarray) shape is (batch_size, 1)
            next_share_obs: (np.ndarray) shape is (batch_size, dim)
            next_actions: (np.ndarray) shape is (n_agents, batch_size, dim)
            gamma: (np.ndarray) shape is (batch_size, 1)
        """
        assert share_obs.__class__.__name__ == "ndarray"
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert next_share_obs.__class__.__name__ == "ndarray"
        assert gamma.__class__.__name__ == "ndarray"
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv_a)
        action = self.indiv_to_joint(actions).to(**self.tpdv_a)
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        next_action = self.indiv_to_joint(next_actions).to(**self.tpdv_a)
        next_q_values = torch.gather(self.target_critic(next_share_obs), 1, next_action)
        if self.use_proper_time_limits:
            q_targets = reward + gamma * next_q_values * (1 - term)
        else:
            q_targets = reward + gamma * next_q_values * (1 - done)
        critic_loss = torch.mean(
            torch.nn.functional.mse_loss(
                torch.gather(self.critic(share_obs), 1, action), q_targets
            )
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def process_action_spaces(self, action_spaces):
        """Process action spaces."""
        self.action_dims = []
        self.joint_action_dim = 1
        for space in action_spaces:
            self.action_dims.append(space.n)
            self.joint_action_dim *= space.n

    def joint_to_indiv(self, orig_action):
        """Convert joint action to individual actions.
        Args:
            orig_action: (int) joint action.
        Returns:
            actions: (list) individual actions.

        For example, if agents' action_dims are [4, 3],
        then:
        joint action 0 <--> indiv actions [0, 0],
        joint action 1 <--> indiv actions [1, 0],
        ......
        joint action 5 <--> indiv actions [1, 1],
        ......
        joint action 11 <--> indiv actions [3, 2].
        """
        action = deepcopy(orig_action)
        actions = []
        for dim in self.action_dims:
            actions.append(action % dim)
            action = torch.div(action, dim, rounding_mode="floor")
        return actions

    def indiv_to_joint(self, orig_actions):
        """Convert individual actions to joint action.
        Args:
            orig_action: (int) joint action.
        Returns:
            actions: (list) individual actions.

        For example, if agents' action_dims are [4, 3],
        then:
        joint action 0 <--> indiv actions [0, 0],
        joint action 1 <--> indiv actions [0, 1],
        ......
        joint action 5 <--> indiv actions [1, 2],
        ......
        joint action 11 <--> indiv actions [3, 2].
        """
        actions = deepcopy(orig_actions)
        action = torch.zeros_like(actions[0])
        accum_dim = 1
        for i, dim in enumerate(self.action_dims):
            action += accum_dim * actions[i]
            accum_dim *= dim
        return action

    def get_joint_idx(self, actions, agent_id):
        """Get available joint idx for an agent.
        All other agents keep their current actions, and this agent can freely choose.
        Args:
            actions: (list) individual actions.
            agent_id: (int) agent id.
        Returns:
            joint_idx: (torch.Tensor) shape is (batch_size, self.action_dims[agent_id])
        """
        batch_size = actions[0].shape[0]
        joint_idx = torch.zeros((batch_size, self.action_dims[agent_id])).to(
            **self.tpdv_a
        )
        accum_dim = 1
        for i, dim in enumerate(self.action_dims):
            if i == agent_id:
                for j in range(self.action_dims[agent_id]):
                    joint_idx[:, j] += accum_dim * j
            else:
                joint_idx += accum_dim * actions[i]
            accum_dim *= dim
        return joint_idx

    def save(self, save_dir):
        """Save model parameters."""
        torch.save(self.critic.state_dict(), str(save_dir) + "/critic_agent" + ".pt")
        torch.save(
            self.target_critic.state_dict(),
            str(save_dir) + "/target_critic_agent" + ".pt",
        )

    def restore(self, model_dir):
        """Restore model parameters."""
        critic_state_dict = torch.load(str(model_dir) + "/critic_agent" + ".pt")
        self.critic.load_state_dict(critic_state_dict)
        target_critic_state_dict = torch.load(
            str(model_dir) + "/target_critic_agent" + ".pt"
        )
        self.target_critic.load_state_dict(target_critic_state_dict)

    def turn_on_grad(self):
        """Turn on gradient for critic."""
        for param in self.critic.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """Turn off gradient for critic."""
        for param in self.critic.parameters():
            param.requires_grad = False
