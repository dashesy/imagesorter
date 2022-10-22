import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class Featurizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Scale((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.ToTensor()
        ])
        model = models.resnet18(pretrained=True).eval()
        layer = model._modules.get('avgpool')

    def preprocess(path: str):

