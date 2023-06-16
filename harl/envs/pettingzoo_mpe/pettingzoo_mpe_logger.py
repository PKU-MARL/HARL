from harl.common.base_logger import BaseLogger


class PettingZooMPELogger(BaseLogger):
    def get_task_name(self):
        if self.env_args["continuous_actions"]:
            return f"{self.env_args['scenario']}-continuous"
        else:
            return f"{self.env_args['scenario']}-discrete"
