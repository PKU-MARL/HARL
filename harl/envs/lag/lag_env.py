import numpy as np

from harl.envs.lag.JSBSim.envs import (
    SingleCombatEnv,
    SingleControlEnv,
    MultipleCombatEnv,
)
import torch


class LAGEnv:
    def __init__(self, env_args):
        self.env_args = env_args
        self.env = self.get_env(env_args)
        self.n_agents = self.env.num_agents
        if self.n_agents == 1:
            self.share_observation_space = [self.env.observation_space]
            self.observation_space = [self.env.observation_space]
            self.action_space = [self.env.action_space]
        else:
            self.share_observation_space = self.repeat(self.env.share_observation_space)
            self.observation_space = self.repeat(self.env.observation_space)
            self.action_space = self.repeat(self.env.action_space)

    def step(self, actions):
        if self.n_agents == 1:
            obs, reward, done, info = self.env.step(actions)
            return obs, obs, reward, done[0], [info], None
        else:
            # obs: (n_agents, obs_dim)
            # share_obs: (n_agents, share_obs_dim)
            # rewards: (n_agents, 1)
            # dones: (n_agents,)
            # infos: (n_agents,)
            # available_actions: None or (n_agents, action_number)
            obs, share_obs, reward, done, info = self.env.step(actions)
            return obs, share_obs, reward, np.squeeze(done), self.repeat(info), None

    def reset(self):
        if self.n_agents == 1:
            obs = self.env.reset()
            return obs, obs, None
        else:
            obs, share_obs = self.env.reset()
            return obs, share_obs, None

    def seed(self, seed):
        pass

    def render(self):
        self.env.render(mode="txt", filepath="render.txt.acmi")

    def close(self):
        self.env.close()

    def get_env(self, env_args):
        if env_args["scenario"] == "SingleCombat":
            env = SingleCombatEnv(env_args["task"])
        elif env_args["scenario"] == "SingleControl":
            env = SingleControlEnv(env_args["task"])
        elif env_args["scenario"] == "MultipleCombat":
            env = MultipleCombatEnv(env_args["task"])
        else:
            print("Can not support the " + env_args["scenario"] + "environment.")
            raise NotImplementedError
        return env

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
