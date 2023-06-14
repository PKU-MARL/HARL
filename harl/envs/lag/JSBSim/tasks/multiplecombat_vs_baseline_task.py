import numpy as np
from gym import spaces
from typing import Tuple
import torch

from .multiplecombat_task import MultipleCombatTask, HierarchicalMultipleCombatTask, HierarchicalMultipleCombatShootTask
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward, MissilePostureReward
from ..utils.utils import get_AO_TA_R, LLA2NEU
from ..core.simulatior import MissileSimulator
from ..tasks import SingleCombatTask


class MultipleCombatVsBaselineTask(MultipleCombatTask):

    @property
    def num_agents(self) -> int:  # ally number
        agent_num = 0
        for key in self.config.aircraft_configs.keys():
            if "A" in key:
                agent_num += 1
        return agent_num

    def load_observation_space(self):
        aircraft_num = len(self.config.aircraft_configs)
        self.obs_length = 9 + (aircraft_num - 1) * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(aircraft_num * self.obs_length,))

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        else:
            norm_act = np.zeros(4)
            norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
            norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
            norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
            norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
            return norm_act


class HierarchicalMultipleCombatVsBaselineTask(HierarchicalMultipleCombatTask):

    @property
    def num_agents(self) -> int:  # ally number
        agent_num = 0
        for key in self.config.aircraft_configs.keys():
            if "A" in key:
                agent_num += 1
        return agent_num

    def load_observation_space(self):
        aircraft_num = len(self.config.aircraft_configs)
        self.obs_length = 9 + (aircraft_num - 1) * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(aircraft_num * self.obs_length,))

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        else:
            # generate low-level input_obs
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)
            # (1) delta altitude/heading/velocity
            input_obs[0] = self.norm_delta_altitude[action[0]]
            input_obs[1] = self.norm_delta_heading[action[1]]
            input_obs[2] = self.norm_delta_velocity[action[2]]
            # (2) ego info
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.expand_dims(input_obs, axis=0)
            # output low-level action
            _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
            action = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            # normalize low-level action
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20 - 1.
            norm_act[1] = action[1] / 20 - 1.
            norm_act[2] = action[2] / 20 - 1.
            norm_act[3] = action[3] / 58 + 0.4
            return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)


class HierarchicalMultipleCombatShootVsBaselineTask(HierarchicalMultipleCombatVsBaselineTask):

    def __init__(self, config: str):
        super().__init__(config)
        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)
        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]

    def load_observation_space(self):
        aircraft_num = len(self.config.aircraft_configs)
        self.obs_length = 9 + aircraft_num * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(aircraft_num * self.obs_length,))

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])

    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        # (1) ego info normalization
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000  # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])  # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])  # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])  # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])  # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340  # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340  # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340  # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340  # 8. ego vc   (unit: mh)(unit: 5G)
        # (2) relative inof w.r.t partner+enemies state
        offset = 8
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset + 1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset + 2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset + 3] = AO
            norm_obs[offset + 4] = TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag
            offset += 6
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        # (3) missile info TODO: multiple missile and parnter's missile?
        missile_sim = env.agents[agent_id].check_missile_warning()  #
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[offset + 1] = (np.linalg.norm(missile_sim.get_velocity()) - ego_state[9]) / 340
            norm_obs[offset + 2] = (missile_feature[2] - ego_state[2]) / 1000
            norm_obs[offset + 3] = ego_AO
            norm_obs[offset + 4] = ego_TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self._remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self._shoot_action = {agent_id: False for agent_id in env.agents.keys()}
        return super().reset(env)

    def normalize_action(self, env, agent_id, action):
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            self._shoot_action[agent_id] = action[3] > 0
            return action
        else:
            self._shoot_action[agent_id] = action[3] > 0
            return super().normalize_action(env, agent_id, action[:3])

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            # Determine whether can launch missile at the nearest enemy aircraft
            target_list = list(map(lambda x: x.get_position() - agent.get_position(), agent.enemies))
            target_distance = list(map(np.linalg.norm, target_list))
            target_index = np.argmin(target_distance)
            target = target_list[target_index]
            heading = agent.get_velocity()
            distance = target_distance[target_index]
            attack_angle = np.rad2deg(
                np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]

            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self._remaining_missiles[agent_id] > 0 \
                         and attack_angle <= self.max_attack_angle and distance <= self.max_attack_distance and shoot_interval >= self.min_attack_interval
            if shoot_flag:
                new_missile_uid = agent_id + str(self._remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[target_index], uid=new_missile_uid))
                self._remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step
