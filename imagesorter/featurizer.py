import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class Featurizer(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, path: str):
        img = Image.open(path).convert("RGB")
        img = self.transforms(img).unsqueeze(0)
        return img

