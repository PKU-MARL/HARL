"""Runner for off-policy HARL algorithms."""
import torch
import numpy as np
import torch.nn.functional as F
from harl.runners.off_policy_base_runner import OffPolicyBaseRunner

class OffPolicyHARunner(OffPolicyBaseRunner):
    """Runner for off-policy HA algorithms."""


    def train(self):
        """Train the model"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # (batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # (batch_size, 1)
            sp_done,  # (batch_size, 1)
            sp_term,  # (batch_size, 1)
            sp_next_share_obs,  # (batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_gamma,  # (batch_size, 1)
        ) = data
        # train critic
        self.critic.turn_on_grad()
        next_actions = []
        for agent_id in range(self.num_agents):
            next_actions.append(self.actor[agent_id].get_target_actions(sp_next_obs[agent_id]))
        self.critic.train(
            sp_share_obs,
            sp_actions,
            sp_reward,
            sp_done,
            sp_term,
            sp_next_share_obs,
            next_actions,
            sp_gamma,
        )
        self.critic.turn_off_grad()
        if self.total_it % self.policy_freq == 0:
            # train actors
            if self.args["algo"] == "had3qn":
                actions = []
                with torch.no_grad():
                    for agent_id in range(self.num_agents):
                        actions.append(self.actor[agent_id].get_actions(sp_obs[agent_id], False))
                # actions shape: (n_agents, batch_size, 1)
                update_actions, get_values = self.critic.train_values(sp_share_obs, actions)
                if self.fixed_order:
                    agent_order = list(range(self.num_agents))
                else:
                    agent_order = list(np.random.permutation(self.num_agents))
                for agent_id in agent_order:
                    self.actor[agent_id].turn_on_grad()
                    # actor preds
                    actor_values = self.actor[agent_id].train_values(
                        sp_obs[agent_id], actions[agent_id]
                    )
                    # critic preds
                    critic_values = get_values()
                    # update
                    actor_loss = torch.mean(F.mse_loss(actor_values, critic_values))
                    self.actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor[agent_id].actor_optimizer.step()
                    self.actor[agent_id].turn_off_grad()
                    update_actions(agent_id)
            else:
                actions = []
                with torch.no_grad():
                    for agent_id in range(self.num_agents):
                        actions.append(self.actor[agent_id].get_actions(sp_obs[agent_id], False))
                # actions shape: (n_agents, batch_size, dim)
                if self.fixed_order:
                    agent_order = list(range(self.num_agents))
                else:
                    agent_order = list(np.random.permutation(self.num_agents))
                for agent_id in agent_order:
                    self.actor[agent_id].turn_on_grad()
                    # train this agent
                    actions[agent_id] = self.actor[agent_id].get_actions(sp_obs[agent_id], False)
                    actions_t = torch.cat(actions, dim=-1)
                    value_pred = self.critic.get_values(sp_share_obs, actions_t)
                    actor_loss = -torch.mean(value_pred)
                    self.actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor[agent_id].actor_optimizer.step()
                    self.actor[agent_id].turn_off_grad()
                    actions[agent_id] = self.actor[agent_id].get_actions(sp_obs[agent_id], False)
            # soft update
            for agent_id in range(self.num_agents):
                self.actor[agent_id].soft_update()
            self.critic.soft_update()
