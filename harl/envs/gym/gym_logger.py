from harl.common.base_logger import BaseLogger


class GYMLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["scenario"]
