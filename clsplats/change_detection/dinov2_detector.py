import torch
import omegaconf

import clsplats.change_detection.base_detector as base_detector
from clsplats.utils.types import Image

class DinoV2Detector(base_detector.BaseDetector):

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        if self.cfg.upsample:
            pass
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.cos = torch.nn.CosineSimilarity(dim=1)

    def _preprocess_image(self, image: Image) -> torch.Tensor:
        pass

    def predict_change_mask(self, rendered_image: Image, observation: Image) -> torch.Tensor:
        rendered_image = self._preprocess_image(rendered_image)
        observation = self._preprocess_image(observation)

        if self.cfg.dilate_mask:
            pass

        return self.cos(rendered_image, observation) > self.cfg.threshold