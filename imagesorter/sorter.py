import os
import os.path as op
from resnet18_featurizer import Resnet18Featurizer as Featurizer
import time
import math
import random


def sort(srcdir: str) -> list[str]:
    featurizer = Featurizer()
    features = {}
    for path in [p for p in os.listdir(srcdir) if op.splitext(p)[1].lower() in [".jpg", ".png"]]:
        path = op.join(srcdir, path)
        features[path] = featurizer(path)

