import numpy as np
from harl.common.base_logger import BaseLogger


class FootballLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["env_name"]

    def eval_init(self):
        super().eval_init()
        self.eval_episode_cnt = 0
        self.eval_score_cnt = 0

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        self.eval_episode_cnt += 1
        if self.eval_infos[tid][0]["score_reward"] > 0:
            self.eval_score_cnt += 1

    def eval_log(self, eval_episode):
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_score_rate = self.eval_score_cnt / self.eval_episode_cnt
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
            "eval_score_rate": [eval_score_rate],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print(
            "Evaluation average episode reward is {}, evaluation score rate is {}.\n".format(
                eval_avg_rew, eval_score_rate
            )
        )
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew, eval_score_rate]))
            + "\n"
        )
        self.log_file.flush()
