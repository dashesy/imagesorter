import torch
from imagesorter.featurizer import Featurizer
from imagesorter.mobilenet_v3_featurizer import MobileNetV3
from imagesorter.resnet18_featurizer import Resnet18Featurizer

class MobileNetResNet18(Featurizer):
    def __init__(self, feat_name_mobilenet: str = None, feat_name: str = None, size: int = 224):
        super().__init__(size)
        self.featurizer_mobilenet = MobileNetV3(feat_name_mobilenet, size)
        self.featurizer_resnet18 = Resnet18Featurizer(feat_name, size)
    
    def forward(self, path: str):
        return self.featurizer_mobilenet(path), self.featurizer_resnet18(path)
