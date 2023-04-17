from __future__ import absolute_import, division, print_function

import time
from os import replace

import numpy as np
from absl import logging
from smacv2.env import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

logging.set_verbosity(logging.DEBUG)
import os.path as osp
from pathlib import Path
import yaml

from gym.spaces import Box, Discrete


class SMACv2Env:
    def __init__(self, args):
        self.map_config = self.load_map_config(args["map_name"])

    def step(self, actions):
        processed_actions = np.squeeze(actions, axis=1).tolist()
        reward, terminated, info = self.env.step(actions)
        obs = self.env.get_obs()
        state = self.repeat(self.env.get_state())
        rewards = [[reward]] * self.n_agents
        dones = [terminated] * self.n_agents
        if terminated:
            if self.env.env.timeouts > self.timeouts:
                assert (
                    self.env.env.timeouts - self.timeouts == 1
                ), "Change of timeouts unexpected."
                info["bad_transition"] = True
                self.timeouts = self.env.env.timeouts
        infos = [info] * self.n_agents
        avail_actions = self.env.get_avail_actions()
        return obs, state, rewards, dones, infos, avail_actions

    def reset(self):
        self.env.reset()
        obs = self.env.get_obs()
        state = self.repeat(self.env.get_state())
        avail_actions = self.env.get_avail_actions()
        return obs, state, avail_actions

    def seed(self, seed):
        self.env = StarCraftCapabilityEnvWrapper(seed=seed, **self.map_config)
        env_info = self.env.get_env_info()
        n_actions = env_info["n_actions"]
        state_shape = env_info["state_shape"]
        obs_shape = env_info["obs_shape"]
        self.n_agents = env_info["n_agents"]
        self.timeouts = self.env.env.timeouts

        self.share_observation_space = self.repeat(
            Box(low=-np.inf, high=np.inf, shape=(state_shape,))
        )
        self.observation_space = self.repeat(
            Box(low=-np.inf, high=np.inf, shape=(obs_shape,))
        )
        self.action_space = self.repeat(Discrete(n_actions))

    def close(self):
        self.env.close()

    def load_map_config(self, map_name):
        base_path = osp.split(osp.split(osp.dirname(osp.abspath(__file__)))[0])[0]
        map_config_path = (
            Path(base_path)
            / "configs"
            / "envs_cfgs"
            / "smacv2_map_config"
            / f"{map_name}.yaml"
        )
        with open(str(map_config_path), "r", encoding="utf-8") as file:
            map_config = yaml.load(file, Loader=yaml.FullLoader)
        return map_config

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
