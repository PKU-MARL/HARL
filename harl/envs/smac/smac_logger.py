import time
from functools import reduce
import numpy as np
from harl.common.base_logger import BaseLogger


class SMACLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(SMACLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        self.win_key = "won"

    def get_task_name(self):
        return self.env_args["map_name"]

    def init(self, episodes):
        self.start = time.time()
        self.episodes = episodes
        self.episode_lens = []
        self.one_episode_len = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.int
        )
        self.last_battles_game = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.float32
        )
        self.last_battles_won = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.float32
        )

    def per_step(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        self.infos = infos
        self.one_episode_len += 1
        done_env = np.all(dones, axis=1)
        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            if done_env[i]:
                self.episode_lens.append(self.one_episode_len[i].copy())
                self.one_episode_len[i] = 0

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()
        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        battles_won = []
        battles_game = []
        incre_battles_won = []
        incre_battles_game = []

        for i, info in enumerate(self.infos):
            if "battles_won" in info[0].keys():
                battles_won.append(info[0]["battles_won"])
                incre_battles_won.append(
                    info[0]["battles_won"] - self.last_battles_won[i]
                )
            if "battles_game" in info[0].keys():
                battles_game.append(info[0]["battles_game"])
                incre_battles_game.append(
                    info[0]["battles_game"] - self.last_battles_game[i]
                )

        incre_win_rate = (
            np.sum(incre_battles_won) / np.sum(incre_battles_game)
            if np.sum(incre_battles_game) > 0
            else 0.0
        )
        self.writter.add_scalars(
            "incre_win_rate", {"incre_win_rate": incre_win_rate}, self.total_num_steps
        )

        self.last_battles_game = battles_game
        self.last_battles_won = battles_won

        average_episode_len = (
            np.mean(self.episode_lens) if len(self.episode_lens) > 0 else 0.0
        )
        self.episode_lens = []

        self.writter.add_scalars(
            "average_episode_length",
            {"average_episode_length": average_episode_len},
            self.total_num_steps,
        )

        for agent_id in range(self.num_agents):
            actor_train_infos[agent_id]["dead_ratio"] = 1 - actor_buffer[
                agent_id
            ].active_masks.sum() / (
                self.num_agents
                * reduce(
                    lambda x, y: x * y, list(actor_buffer[agent_id].active_masks.shape)
                )
            )

        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)

        print(
            "Increase games {:.4f}, win rate on these games is {:.4f}, average step reward is {:.4f}, average episode length is {:.4f}, average episode reward is {:.4f}.\n".format(
                np.sum(incre_battles_game),
                incre_win_rate,
                critic_train_info["average_step_rewards"],
                average_episode_len,
                average_episode_len * critic_train_info["average_step_rewards"],
            )
        )

    def eval_init(self):
        super().eval_init()
        self.eval_battles_won = 0

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        if self.eval_infos[tid][0][self.win_key] == True:
            self.eval_battles_won += 1

    def eval_log(self, eval_episode):
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_win_rate = self.eval_battles_won / eval_episode
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
            "eval_win_rate": [eval_win_rate],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print(
            "Evaluation win rate is {}, evaluation average episode reward is {}.\n".format(
                eval_win_rate, eval_avg_rew
            )
        )
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew, eval_win_rate]))
            + "\n"
        )
        self.log_file.flush()
