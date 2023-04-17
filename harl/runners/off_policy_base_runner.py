"""Base runner for off-policy algorithms."""
import os
import time
import torch
import numpy as np
import setproctitle
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import (
    init_dir,
    save_config,
    get_task_name
)
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics import CRITIC_REGISTRY
from harl.common.buffers.off_policy_buffer import OffPolicyBuffer

class OffPolicyBaseRunner:
    """Base runner for off-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OffPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        if "policy_freq" in self.algo_args["algo"]:
            self.policy_freq = self.algo_args["algo"]["policy_freq"]
        else:
            self.policy_freq = 1

        self.share_param = algo_args["algo"]['share_param']
        self.fixed_order = algo_args["algo"]['fixed_order']

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.task_name = get_task_name(args["env"], env_args)
        if not self.algo_args['render']['use_render']:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)
            self.log_file = open(os.path.join(self.run_dir, "progress.txt"), "w", encoding='utf-8')
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # env
        if self.algo_args['render']['use_render']:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        self.action_spaces = self.envs.action_space
        for agent_id in range(self.num_agents):
            self.action_spaces[agent_id].seed(algo_args["seed"]["seed"] + agent_id + 1)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        if self.share_param:
            self.actor = []
            agent = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            self.actor.append(agent)
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id] == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                self.actor.append(agent)

        if not self.algo_args['render']['use_render']:
            self.critic = CRITIC_REGISTRY[args["algo"]](
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                self.envs.share_observation_space[0],
                self.envs.action_space,
                device=self.device,
            )
            self.buffer = OffPolicyBuffer(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                self.envs.share_observation_space[0],
                self.num_agents,
                self.envs.observation_space,
                self.envs.action_space,
            )

        if self.algo_args['train']['model_dir'] is not None:
            self.restore()

        self.total_it = 0  # total iteration

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args['render']['use_render']:  # render, not train
            self.render()
            return
        # warmup
        print("start warmup")
        obs, share_obs = self.warmup()
        print("finish warmup, start training")
        # train and eval
        steps = (
            self.algo_args['train']['num_env_steps'] // self.algo_args['train']['n_rollout_threads']
        )
        update_num = int(  # update number per train
            self.algo_args['train']['update_per_train'] * self.algo_args['train']['train_interval']
        )
        for step in range(1, steps + 1):
            actions = self.get_actions(obs, add_random=True)
            new_obs, new_share_obs, reward, done, infos, _ = self.envs.step(
                actions
            )  # reward: (n_threads, n_agents, 1); done: (n_threads, n_agents)
            terms = np.full((self.algo_args['train']['n_rollout_threads'], 1), False)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            for i in range(self.algo_args['train']['n_rollout_threads']):
                if done[i][0]:
                    if not (
                        "bad_transition" in infos[i][0].keys()
                        and infos[i][0]["bad_transition"] == True
                    ):  # not trunc
                        terms[i][0] = True
                    next_obs[i] = infos[i][0]["original_obs"].copy()
                    next_share_obs[i] = infos[i][0]["original_state"].copy()
            reward = reward[:, 0]
            done = np.expand_dims(np.all(done, axis=1), axis=-1)
            data = (
                share_obs[:, 0],
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                reward,
                done,
                terms,
                next_share_obs[:, 0],
                next_obs.transpose(1, 0, 2),
            )
            self.buffer.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            if step % self.algo_args['train']['train_interval'] == 0:
                if self.algo_args['train']['use_linear_lr_decay']:
                    if self.share_param:
                        self.actor[0].lr_decay(step, steps)
                    else:
                        for agent_id in range(self.num_agents):
                            self.actor[agent_id].lr_decay(step, steps)
                    self.critic.lr_decay(step, steps)
                for _ in range(update_num):
                    self.train()
            if step % self.algo_args['train']['eval_interval'] == 0:
                cur_step = (
                    self.algo_args['train']['warmup_steps']
                    + step * self.algo_args['train']['n_rollout_threads']
                )
                if self.algo_args['eval']['use_eval']:
                    print(
                        f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Evaluation at step {cur_step}:"
                    )
                    self.eval(cur_step)
                else:
                    print(
                        f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Step {cur_step}, average step reward in buffer: {self.buffer.get_mean_rewards()}.\n"
                    )
                self.save()

    def warmup(self):
        """Warmup the replay buffer with random actions"""
        warmup_steps = (
            self.algo_args['train']['warmup_steps'] // self.algo_args['train']['n_rollout_threads']
        )
        # obs: (n_threads, n_agents, dim)
        # share_obs: (n_threads, n_agents, dim)
        obs, share_obs, _ = self.envs.reset()
        for _ in range(warmup_steps):
            # action: (n_threads, n_agents, dim)
            actions = self.sample_actions()
            new_obs, new_share_obs, reward, done, infos, _ = self.envs.step(actions)
            terms = np.full((self.algo_args['train']['n_rollout_threads'], 1), False)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            for i in range(self.algo_args['train']['n_rollout_threads']):
                if done[i][0]:
                    if not (
                        "bad_transition" in infos[i][0].keys()
                        and infos[i][0]["bad_transition"] == True
                    ):
                        terms[i][0] = True
                    next_obs[i] = infos[i][0]["original_obs"].copy()
                    next_share_obs[i] = infos[i][0]["original_state"].copy()
            data = (
                share_obs[:, 0],
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                reward[:, 0],
                np.expand_dims(done, axis=-1)[:, 0],
                terms,
                next_share_obs[:, 0],
                next_obs.transpose(1, 0, 2),
            )
            self.buffer.insert(data)
            obs = new_obs
            share_obs = new_share_obs
        return obs, share_obs
    
    def sample_actions(self):
        """Sample random actions for warmup.
        Returns:
            actions: (np.ndarray) sampled actions, shape is (n_threads, n_agents, dim)
        """
        actions = []
        for agent_id in range(self.num_agents):
            action = []
            for _ in range(self.algo_args['train']['n_rollout_threads']):
                action.append(self.action_spaces[agent_id].sample())
            actions.append(action)
        if self.envs.action_space[agent_id].__class__.__name__ == "Box":
            return np.array(actions).transpose(1, 0, 2)

        return np.expand_dims(np.array(actions).transpose(1, 0), axis=-1)

    @torch.no_grad()
    def get_actions(self, obs, add_random=True):
        """Get actions for rollout.
        Args:
            obs: (np.ndarray) input observation, shape is (n_threads, n_agents, dim)
            add_random: (bool) whether to add randomness
        Returns:
            actions: (np.ndarray) agent actions, shape is (n_threads, n_agents, dim)
        """
        actions = []
        for agent_id in range(self.num_agents):
            actions.append(_t2n(self.actor[agent_id].get_actions(obs[:, agent_id], add_random)))
        return np.array(actions).transpose(1, 0, 2)

    def train(self):
        """Train the model"""
        raise NotImplementedError

    @torch.no_grad()
    def eval(self, step):
        """Evaluate the model"""
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.algo_args['eval']['n_eval_rollout_threads']):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
        eval_episode = 0
        episode_lens = []
        one_episode_len = np.zeros(self.algo_args['eval']['n_eval_rollout_threads'], dtype=np.int)

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        while True:
            eval_actions = self.get_actions(eval_obs, False)
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            for eval_i in range(self.algo_args['eval']['n_eval_rollout_threads']):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            one_episode_len += 1

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.algo_args['eval']['n_eval_rollout_threads']):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []
                    episode_lens.append(one_episode_len[eval_i].copy())
                    one_episode_len[eval_i] = 0

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                eval_episode_rewards = np.concatenate(
                    [rewards for rewards in eval_episode_rewards if rewards]
                )
                eval_avg_rew = np.mean(eval_episode_rewards)
                eval_avg_len = np.mean(episode_lens)
                print(
                    f'Eval average episode reward is {eval_avg_rew}, eval average episode length is {eval_avg_len}.\n'
                )
                self.log_file.write(",".join(map(str, [step, eval_avg_rew, eval_avg_len])) + "\n")
                self.log_file.flush()
                self.writter.add_scalar("eval_average_episode_rewards", eval_avg_rew, step)
                self.writter.add_scalar("eval_average_episode_length", eval_avg_len, step)
                break

    @torch.no_grad()
    def render(self):
        """Render the model"""
        print("start rendering")
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for _ in range(self.algo_args['render']['render_episodes']):
                eval_obs, _, _ = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                rewards = 0
                while True:
                    eval_actions = self.get_actions(eval_obs, False)
                    eval_obs, _, eval_rewards, eval_dones, _, _ = self.envs.step(eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        print(f'total reward of this episode: {rewards}')
                        break
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            for _ in range(self.algo_args['render']['render_episodes']):
                eval_obs, _, _ = self.envs.reset()
                rewards = 0
                while True:
                    eval_actions = self.get_actions(eval_obs, False)
                    eval_obs, _, eval_rewards, eval_dones, _, _ = self.envs.step(eval_actions)
                    rewards += eval_rewards[0][0][0]
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        if "smac" in self.args["env"]:  # replay for smac, no rendering
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def restore(self):
        """Restore the model"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].restore(self.algo_args['train']['model_dir'], agent_id)
        if not self.algo_args['render']['use_render']:
            self.critic.restore(self.algo_args['train']['model_dir'])

    def save(self):
        """Save the model"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].save(self.save_dir, agent_id)
        self.critic.save(self.save_dir)

    def close(self):
        """Close environment, writter, and log file."""
        # post process
        if self.algo_args['render']['use_render']:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.log_file.close()