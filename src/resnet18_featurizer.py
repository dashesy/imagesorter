import imp
import torchvision.models as models
from featurizer import Featurizer

class Resnet18Featurizer(Featurizer):
    def __init__(self):
        super().__init__()
