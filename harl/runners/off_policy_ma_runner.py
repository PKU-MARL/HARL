"""Runner for off-policy MA algorithms"""
import copy
import torch
import torch.nn.functional as F
from harl.runners.off_policy_base_runner import OffPolicyBaseRunner
from harl.utils.discrete_util import get_onehot_actions


class OffPolicyMARunner(OffPolicyBaseRunner):
    """Runner for off-policy MA algorithms."""

    def train(self):
        """Train the model"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        # train critic
        self.critic.turn_on_grad()
        next_actions = []
        for agent_id in range(self.num_agents):
            next_actions.append(
                self.actor[agent_id].get_target_actions(
                    sp_next_obs[agent_id],
                    sp_next_available_actions[agent_id]
                    if sp_next_available_actions is not None
                    else None,
                )
            )
        self.critic.train(
            sp_share_obs,
            sp_actions,
            sp_reward,
            sp_done,
            sp_valid_transition,
            sp_term,
            sp_next_share_obs,
            next_actions,
            sp_gamma,
            self.value_normalizer,
        )
        self.critic.turn_off_grad()
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)
        if self.total_it % self.policy_freq == 0:
            # train actors
            # actions shape: (n_agents, batch_size, dim)
            for agent_id in range(self.num_agents):
                actions = copy.deepcopy(torch.tensor(sp_actions)).to(self.device)
                if self.action_type != "Box":
                    actions = actions.to(dtype=torch.int64)
                    actions = get_onehot_actions(
                        actions, self.num_agents, self.action_type, self.action_spaces
                    )
                self.actor[agent_id].turn_on_grad()
                # train this agent
                actions[agent_id] = self.actor[agent_id].get_actions(
                    sp_obs[agent_id],
                    sp_available_actions[agent_id]
                    if sp_available_actions is not None
                    else None,
                    add_noise=False,
                    onehot=True,
                )
                actions_list = [a for a in actions]
                if self.state_type == "EP":
                    actions_t = torch.cat(actions_list, dim=-1)
                elif self.state_type == "FP":
                    actions_t = torch.tile(
                        torch.cat(actions_list, dim=-1), (self.num_agents, 1)
                    )
                value_pred = self.critic.get_values(sp_share_obs, actions_t)
                if self.algo_args["algo"]["use_policy_active_masks"]:
                    if self.state_type == "EP":
                        actor_loss = (
                            -torch.sum(value_pred * sp_valid_transition[agent_id])
                            / sp_valid_transition[agent_id].sum()
                        )
                    elif self.state_type == "FP":
                        valid_transition = torch.tile(
                            sp_valid_transition[agent_id], (self.num_agents, 1)
                        )
                        actor_loss = (
                            -torch.sum(value_pred * valid_transition)
                            / valid_transition.sum()
                        )
                else:
                    actor_loss = -torch.mean(value_pred)
                self.actor[agent_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor[agent_id].actor_optimizer.step()
                self.actor[agent_id].turn_off_grad()
            # soft update
            for agent_id in range(self.num_agents):
                self.actor[agent_id].soft_update()
            self.critic.soft_update()
