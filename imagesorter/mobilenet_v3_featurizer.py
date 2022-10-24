import torch
import torchvision.models.detection as models
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
from imagesorter.featurizer import Featurizer

class MobileNetV3(Featurizer):
    def __init__(self, feat_name: str = None, size: int = 512):
        super().__init__()
        weights = models.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = models.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        self.feat_name = feat_name or "fpn"
        self.featurizer = create_feature_extractor(model.backbone, return_nodes=[self.feat_name])
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            weights.transforms(),
        ])
    
    def forward(self, path: str):
        img = super().forward(path)
        feats = self.featurizer(img)[self.feat_name]

        return torch.cat([feats['0'], feats['1']]).flatten()
