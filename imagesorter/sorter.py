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
    elif featurizer.startswith("mobilenet_resnet18"):
        from imagesorter.mobilenet_resnet18_featurizer import MobileNetResNet18 as Featurizer
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

def get_sims(srcdir: str, featurizer:str):
    """Find the similarities between all the pairs,
    Also return the minimum pair
    """
    features = featurize(srcdir, featurizer)
    sim = nn.CosineSimilarity(eps=1e-6, dim=0)
    srcs = list(features.keys())
    sims = {}
    min_sim = None
    min_key = None
    for ii in range(len(srcs) - 1):
        for jj in range(ii, len(srcs)):
            s = sim(features[srcs[ii]], features[srcs[jj]])
            sims[(ii, jj)] = sims[(jj, ii)] = s
            if min_sim is None or s < min_sim:
                min_sim = s
                min_key = (ii, jj)
    return sims, srcs, set(min_key)

def sort(srcdir: str, featurizer:str) -> list[str]:
    """Sort all the images in the path and return the sorted names
    """
    sims, srcs, min_key = get_sims(srcdir, featurizer)
    sorted = []
    # start with the least simmillar, so everything else is sorted accordingly
    for k in min_key:
        sorted.append(k)
    for k in range(len(srcs)):
        if k in min_key:
            continue
        # got through previously sorted, and find the top-2 most similar to k
        # that determines the direction to search where to place k
        max_sim = None
        max_idx = None
        max2_idx = None
        max2_sim = None
        for ii, prev_k in enumerate(sorted):
            s = sims[(k, prev_k)]
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
            if sims[(sorted[ii], sorted[ii + step])] < sims[(sorted[ii], k)]:
                sorted.insert((ii + step) if step > 0 else ii, k)
                found = True
                break
        if not found:
            sorted.insert(0 if step < 0 else len(sorted), k)
    
    return [srcs[k] for k in sorted]
