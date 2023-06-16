"""Tools for HARL."""
import os
import random
import numpy as np
import torch
from harl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape


def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    if env_name == "dexhands":
        from harl.envs.dexhands.dexhands_env import DexHandsEnv

        return DexHandsEnv({"n_threads": n_threads, **env_args})

    def get_env_fn(rank):
        def init_env():
            if env_name == "smac":
                from harl.envs.smac.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(env_args)
            elif env_name == "smacv2":
                from harl.envs.smacv2.smacv2_env import SMACv2Env

                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import (
                    MujocoMulti,
                )

                env = MujocoMulti(env_args=env_args)
            elif env_name == "pettingzoo_mpe":
                from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import (
                    PettingZooMPEEnv,
                )

                assert env_args["scenario"] in [
                    "simple_v2",
                    "simple_spread_v2",
                    "simple_reference_v2",
                    "simple_speaker_listener_v3",
                ], "only cooperative scenarios in MPE are supported"
                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from harl.envs.gym.gym_env import GYMEnv

                env = GYMEnv(env_args)
            elif env_name == "football":
                from harl.envs.football.football_env import FootballEnv

                env = FootballEnv(env_args)
            elif env_name == "lag":
                from harl.envs.lag.lag_env import LAGEnv

                env = LAGEnv(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    if env_name == "dexhands":  # dexhands does not support running multiple instances
        raise NotImplementedError

    def get_env_fn(rank):
        def init_env():
            if env_name == "smac":
                from harl.envs.smac.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(env_args)
            elif env_name == "smacv2":
                from harl.envs.smacv2.smacv2_env import SMACv2Env

                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import (
                    MujocoMulti,
                )

                env = MujocoMulti(env_args=env_args)
            elif env_name == "pettingzoo_mpe":
                from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import (
                    PettingZooMPEEnv,
                )

                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from harl.envs.gym.gym_env import GYMEnv

                env = GYMEnv(env_args)
            elif env_name == "football":
                from harl.envs.football.football_env import FootballEnv

                env = FootballEnv(env_args)
            elif env_name == "lag":
                from harl.envs.lag.lag_env import LAGEnv

                env = LAGEnv(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    manual_render = True  # manually call the render() function
    manual_expand_dims = True  # manually expand the num_of_parallel_envs dimension
    manual_delay = True  # manually delay the rendering by time.sleep()
    env_num = 1  # number of parallel envs
    if env_name == "smac":
        from harl.envs.smac.StarCraft2_Env import StarCraft2Env

        env = StarCraft2Env(args=env_args)
        manual_render = (
            False  # smac does not support manually calling the render() function
        )
        # instead, it use save_replay()
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "smacv2":
        from harl.envs.smacv2.smacv2_env import SMACv2Env

        env = SMACv2Env(args=env_args)
        manual_render = False
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "mamujoco":
        from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

        env = MujocoMulti(env_args=env_args)
        env.seed(seed * 60000)
    elif env_name == "pettingzoo_mpe":
        from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv

        env = PettingZooMPEEnv({**env_args, "render_mode": "human"})
        env.seed(seed * 60000)
    elif env_name == "gym":
        from harl.envs.gym.gym_env import GYMEnv

        env = GYMEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "football":
        from harl.envs.football.football_env import FootballEnv

        env = FootballEnv(env_args)
        manual_render = False  # football renders automatically
        env.seed(seed * 60000)
    elif env_name == "dexhands":
        from harl.envs.dexhands.dexhands_env import DexHandsEnv

        env = DexHandsEnv({"n_threads": 64, **env_args})
        manual_render = False  # dexhands renders automatically
        manual_expand_dims = (
            False  # dexhands uses parallel envs, thus dimension is already expanded
        )
        manual_delay = False
        env_num = 64
    elif env_name == "lag":
        from harl.envs.lag.lag_env import LAGEnv

        env = LAGEnv(env_args)
        env.seed(seed * 60000)
    else:
        print("Can not support the " + env_name + "environment.")
        raise NotImplementedError
    return env, manual_render, manual_expand_dims, manual_delay, env_num


def set_seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if env == "smac":
        from harl.envs.smac.smac_maps import get_map_params

        return get_map_params(env_args["map_name"])["n_agents"]
    elif env == "smacv2":
        return envs.n_agents
    elif env == "mamujoco":
        return envs.n_agents
    elif env == "pettingzoo_mpe":
        return envs.n_agents
    elif env == "gym":
        return envs.n_agents
    elif env == "football":
        return envs.n_agents
    elif env == "dexhands":
        return envs.n_agents
    elif env == "lag":
        return envs.n_agents
