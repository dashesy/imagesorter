import torch
import torch.nn as nn
import os
import os.path as op

def featurize(srcdir: str, featurizer:str):
    """Featurize all the images in the path
    """
    feat_name = None
    if featurizer.startswith("vit_b_16"):
        from imagesorter.vit_b_16_featurizer import ViTB16Featurizer as Featurizer
        if featurizer == "vit_b_16:getitem_5":
            feat_name = "getitem_5"
    elif featurizer.startswith("mobilenet"):
        from imagesorter.mobilenet_v3_featurizer import MobileNetV3 as Featurizer
    else:
        assert featurizer == "resnet18"
        from imagesorter.resnet18_featurizer import Resnet18Featurizer as Featurizer
    featurizer = Featurizer(feat_name=feat_name).eval()
    features = {}
    with torch.no_grad():
        for path in [p for p in os.listdir(srcdir) if op.splitext(p)[1].lower() in [".jpg", ".png"]]:
            path = op.join(srcdir, path)
            features[path] = featurizer(path)
    return features

def sort(srcdir: str, featurizer:str) -> list[str]:
    """Sort all the images in the path and return the sorted names
    """
    features = featurize(srcdir, featurizer)
    sim = nn.CosineSimilarity(eps=1e-6, dim=0)
    sorted = []
    sims = {}
    for src, feat in features.items():
        if len(sorted) < 2:
            # add the first two
            sorted.append(src)
            continue
        if len(sorted) == 2:
            sims[(sorted[0], sorted[1])] = sims[(sorted[1], sorted[0])] = sim(features[sorted[0]], features[sorted[1]])
        # got through previously sorted, and find the top-2 most similar
        max_sim = None
        max_idx = None
        max2_idx = None
        max2_sim = None
        for ii, prev_src in enumerate(sorted):
            s = sim(feat, features[prev_src])
            sims[(src, prev_src)] = sims[(prev_src, src)] = s
            if max_sim is None or s > max_sim:
                max2_idx = max_idx
                max_sim = s
                max_idx = ii
            elif max2_sim is None or s > max2_sim:
                max2_sim = s
                max2_idx = ii
        assert max2_idx is not None
        found = False
        step = 1 if max2_idx > max_idx else -1
        max2_idx = 0 if step < 0 else len(sorted) - 1
        for ii in range(max_idx, max2_idx, step):
            if sims[(sorted[ii], sorted[ii + step])] < sims[(sorted[ii], src)]:
                sorted.insert((ii + step) if step > 0 else ii, src)
                found = True
                break
        if not found:
            sorted.insert(0 if step < 0 else len(sorted), src)
    
    return sorted
