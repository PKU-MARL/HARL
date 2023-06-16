from harl.envs.smac.smac_logger import SMACLogger


class SMACv2Logger(SMACLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(SMACv2Logger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        self.win_key = "battle_won"
