import torch

from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class YuNetDeploy(SingleStageDetector):
    def __init__(
        self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None, pretrained=None
    ):
        super(YuNetDeploy, self).__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def forward_train(
        self,
        img: torch.Tensor,
    ) -> torch.Tensor:
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs
