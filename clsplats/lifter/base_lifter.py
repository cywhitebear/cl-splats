import abc

import omegaconf

class BaseLifter(abc.ABC):

    def __init__(self, cfg: omegaconf.DictConfig):
        self.cfg = cfg

    @abc.abstractmethod
    def lift(self, rendered_image: Image, observation: Image) -> torch.Tensor: