import copy
import numpy as np
import gfootball.env as football_env
from gym.spaces import Discrete, Box

env_num_agents = {
    "academy_pass_and_shoot_with_keeper": 2,
    "academy_run_pass_and_shoot_with_keeper": 2,
    "academy_3_vs_1_with_keeper": 3,
    "academy_counterattack_easy": 4,
    "academy_counterattack_hard": 4,
    "academy_corner": 11,
    "academy_single_goal_versus_lazy": 11,
}


class FootballEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.process_args(self.args)
        self.env = football_env.create_environment(**self.args)
        self.n_agents = env_num_agents[self.args["env_name"]]
        self.share_observation_space = self.repeat(self.get_state_space())
        self.observation_space = self.get_obs_shape()
        self.action_space = [Discrete(i) for i in self.env.action_space.nvec]
        self.avail_actions = self.get_avail_actions()

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        obs, rew, done, info = self.env.step(actions.flatten())
        rewards = [[rew[0]]] * self.n_agents
        if self.img:
            obs = obs.transpose(0, 3, 1, 2)
        return (
            self.split(obs),
            self.repeat(self.get_state()),
            rewards,
            self.repeat(done),
            self.repeat(info),
            self.avail_actions,
        )

    def reset(self):
        """Returns initial observations and states"""
        obs = self.env.reset()
        if self.img:
            obs = obs.transpose(0, 3, 1, 2)
        return self.split(obs), self.repeat(self.get_state()), self.avail_actions

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)

    def process_args(self, args):
        if "channel_dimensions" in args:
            args["channel_dimensions"] = tuple(args["channel_dimensions"])
        if "logdir" in args and args["logdir"] is None:
            args["logdir"] = ""
        if "other_config_options" in args and args["other_config_options"] is None:
            args["other_config_options"] = {}
        if self.args["representation"] in ("pixels", "pixels_gray", "extracted"):
            self.img = True
        else:
            self.img = False

    def get_state_space(self):
        # state space is designed following Simple115StateWrapper.convert_observation
        # global states are included once, and the active one-hot encodings for all players are included.
        total_length = 115 + (self.n_agents - 1) * 11
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_length,),
            dtype=self.env.observation_space.dtype,
        )

    def get_state(self):
        # adapted from imple115StateWrapper.convert_observation
        raw_state = self.env.unwrapped.observation()

        def do_flatten(obj):
            """Run flatten on either python list or numpy array."""
            if type(obj) == list:
                return np.array(obj).flatten()
            return obj.flatten()

        s = []
        for i, name in enumerate(
            ["left_team", "left_team_direction", "right_team", "right_team_direction"]
        ):
            s.extend(do_flatten(raw_state[0][name]))
            # If there were less than 11vs11 players we backfill missing values
            # with -1.
            if len(s) < (i + 1) * 22:
                s.extend([-1] * ((i + 1) * 22 - len(s)))
        # ball position
        s.extend(raw_state[0]["ball"])
        # ball direction
        s.extend(raw_state[0]["ball_direction"])
        # one hot encoding of which team owns the ball
        if raw_state[0]["ball_owned_team"] == -1:
            s.extend([1, 0, 0])
        if raw_state[0]["ball_owned_team"] == 0:
            s.extend([0, 1, 0])
        if raw_state[0]["ball_owned_team"] == 1:
            s.extend([0, 0, 1])
        game_mode = [0] * 7
        game_mode[raw_state[0]["game_mode"]] = 1
        s.extend(game_mode)
        for obs in raw_state:
            active = [0] * 11
            if obs["active"] != -1:
                active[obs["active"]] = 1
            s.extend(active)
        return np.array(s, dtype=np.float32)

    def get_obs_shape(self):
        obs_sp = self.env.observation_space
        if self.img:
            w, h, c = self.env.observation_space.shape[1:]
            return [
                Box(
                    low=obs_sp.low[idx].transpose(2, 0, 1),
                    high=obs_sp.high[idx].transpose(2, 0, 1),
                    shape=(c, w, h),
                    dtype=obs_sp.dtype,
                )
                for idx in range(self.n_agents)
            ]
        else:
            return [
                Box(
                    low=obs_sp.low[idx],
                    high=obs_sp.high[idx],
                    shape=obs_sp.shape[1:],
                    dtype=obs_sp.dtype,
                )
                for idx in range(self.n_agents)
            ]

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

    def split(self, a):
        return [a[i] for i in range(self.n_agents)]
