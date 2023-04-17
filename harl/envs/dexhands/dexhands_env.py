import torch
from harl.envs.dexhands.DexterousHands.bidexhands.utils.config import (
    get_args,
    load_env_cfg,
    parse_sim_params,
)
from harl.envs.dexhands.DexterousHands.bidexhands.utils.process_marl import (
    get_AgentIndex,
)
from harl.envs.dexhands.DexterousHands.bidexhands.utils.parse_task import (
    parse_task,
)


def _t2n(x):
    return x.detach().cpu().numpy()


class DexHandsEnv:
    def __init__(self, env_args):
        self.env_args = env_args
        self.env = self.get_env(env_args)
        self.n_envs = env_args["n_threads"]
        self.n_agents = self.env.num_agents
        self.share_observation_space = self.env.share_observation_space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, actions):
        actions = torch.tensor(actions.transpose(1, 0, 2))
        obs_all, state_all, reward_all, done_all, info_all, _ = self.env.step(actions)
        return (
            _t2n(obs_all),
            _t2n(state_all),
            _t2n(reward_all),
            _t2n(done_all),
            [[{}, {}]] * self.n_envs,
            [None] * self.env_args["n_threads"],
        )

    def reset(self):
        obs, s_obs, _ = self.env.reset()
        return _t2n(obs), _t2n(s_obs), [None] * self.env_args["n_threads"]

    def close(self):
        pass

    def get_env(self, env_args):
        # env_args.keys(): ["n_threads", "hands_episode_length", "task"]
        args = get_args(env_args)
        cfg = load_env_cfg(args)
        sim_params = parse_sim_params(args, cfg)
        agent_index = get_AgentIndex(cfg)
        args.task_type = "MultiAgent"
        env = parse_task(args, cfg, sim_params, agent_index)
        return env
