
class CLSplatsTrainer:

    def __init__(self, cfg: omegaconf.DictConfig):
        self.cfg = cfg
        self.timestep = 0
    
    def _pre_step(self):
        pass

    def _post_step(self):
        pass

    def _train_step(self):
        pass

    def train(self):
        pass

    def prepare_timestep(self, timestep: int):
        assert timestep < self.cfg.train.num_times, "Timestep must be less than num_times"
        self.timestep = timestep

    def log_history(self):
        pass