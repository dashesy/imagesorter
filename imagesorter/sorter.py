import imp
import torch
import torch.nn as nn
import os
import os.path as op
from imagesorter.resnet18_featurizer import Resnet18Featurizer as Featurizer

def featurize(srcdir: str):
    featurizer = Featurizer().eval()
    features = {}
    with torch.no_grad():
        for path in [p for p in os.listdir(srcdir) if op.splitext(p)[1].lower() in [".jpg", ".png"]]:
            path = op.join(srcdir, path)
            features[path] = featurizer(path)
    return features

def sort(srcdir: str) -> list[str]:
    features = featurize(srcdir)
    sorted = []
    sim = nn.CosineSimilarity(eps=1e-6, dim=0)

