import copy
import gym


class GYMEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.env = gym.make(args["scenario"])
        self.n_agents = 1
        self.share_observation_space = [self.env.observation_space]
        self.observation_space = [self.env.observation_space]
        self.action_space = [self.env.action_space]
        if self.env.action_space.__class__.__name__ == "Box":
            self.discrete = False
        else:
            self.discrete = True

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        if self.discrete:
            obs, rew, done, info = self.env.step(actions.flatten()[0])
        else:
            obs, rew, done, info = self.env.step(actions[0])
        if done:
            if (
                "TimeLimit.truncated" in info.keys()
                and info["TimeLimit.truncated"] == True
            ):
                info["bad_transition"] = True
        return [obs], [obs], [[rew]], [done], [info], self.get_avail_actions()

    def reset(self):
        """Returns initial observations and states"""
        obs = [self.env.reset()]
        s_obs = copy.deepcopy(obs)
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = [[1] * self.action_space[0].n]
            return avail_actions
        else:
            return None

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)
