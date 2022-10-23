import imp
import torch
import torch.nn as nn
import os
import os.path as op
from imagesorter.resnet18_featurizer import Resnet18Featurizer as Featurizer

def featurize(srcdir: str):
    """Featurize all the images in the path
    """
    featurizer = Featurizer().eval()
    features = {}
    with torch.no_grad():
        for path in [p for p in os.listdir(srcdir) if op.splitext(p)[1].lower() in [".jpg", ".png"]]:
            path = op.join(srcdir, path)
            features[path] = featurizer(path)
    return features

def sort(srcdir: str) -> list[str]:
    """Sort all the images in the path and return the sorted names
    """
    features = featurize(srcdir)
    sim = nn.CosineSimilarity(eps=1e-6, dim=0)
    sorted = []
    sims = {}
    for src, feat in features.items():
        if len(sorted) < 2:
            # add the first two
            sorted.append(src)
            continue
        # got through previously sorted, and find the top-2 most similar
        max_sim = None
        max_idx = None
        max2_idx = None
        for ii, prev_src in enumerate(sorted):
            s = sim(feat, features[prev_src])
            sims[(src, prev_src)] = sims[(prev_src, src)] = s
            if max_sim is None or s > max_sim:
                max2_idx = max_idx
                max_sim = s
                max_idx = ii
        step = 1 if max2_idx > max_idx else -1
        for ii in range(max_idx + step, max2_idx, step):
            if sims[(sorted[ii], )] < max_sim:
                sorted.insert(ii, src)
                break
        
    
    return sorted
