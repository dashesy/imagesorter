import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from imagesorter.featurizer import Featurizer

class ViTB16Featurizer(Featurizer):
    def __init__(self, size: int = 224, feat_name: str = None):
        super().__init__(size)
        model = models.vit_b_16(pretrained=True)
        self.feat_name = feat_name or "encoder.ln"
        self.featurizer = create_feature_extractor(model, return_nodes=[self.feat_name])
    
    def forward(self, path: str):
        img = super().forward(path)
        return self.featurizer(img)[self.feat_name].flatten()
