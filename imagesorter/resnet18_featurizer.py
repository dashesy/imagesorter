import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from featurizer import Featurizer

class Resnet18Featurizer(Featurizer):
    def __init__(self, size: int = 112):
        super().__init__(size)
        model = models.resnet18(pretrained=True).eval()
        self.featurizer = create_feature_extractor(model, 'avgpool').eval()
    
    def forward(self, path: str):
        img = super()(path)
        return self.featurizer(img)["avgpool"].squeeze()
