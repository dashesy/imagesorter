import os
import os.path as op
import argparse
import shutil
import imagesorter.sorter as sorter

def main(srcdir: str, dstdir: str, short: bool, featurizer:str, dedup_threshold: float):
    sorted, scores = sorter.sort(srcdir, featurizer=featurizer)
    if not sorted:
        return
    os.makedirs(dstdir, exist_ok=True)
    length = len(str(len(sorted)))
    srcs = []
    cnt = 0
    with open(op.join(dstdir, "sorted.csv"), "w") as wfp:
        for ii in range(len(sorted)):
            src = sorted[ii]
            score = scores[ii]
            srcs.append(src)
            if score > dedup_threshold:
                continue
            ext = op.splitext(src)[1]
            if not short:
                orig = [op.splitext(op.basename(src))[0] for src in srcs]
                orig.sort()
                ext = "-" + "-".join(orig) + ext
            dst = str(cnt).rjust(length, '0') + ext
            for src in srcs:
                print(f"{src},{dst}", file=wfp)
            shutil.copy(src, op.join(dstdir, dst))
            cnt += 1
            srcs = []
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sort input images into output directory with numerical names')
    parser.add_argument('src', help='path to input directory of images to be sorted')
    parser.add_argument('dst', help='path to output directory to add sorted images')
    parser.add_argument('-s', '--short', action='store_true', help='only keep the short numerical sorted file names')
    parser.add_argument('-d', '--dedup', type=float, default=2.0, help='deduplication threshold (2.0 default for no deduplication)')
    parser.add_argument('-f', '--featurizer', choices=['mobilenet+resnet18', 'resnet18', 'vit_b_16', 'vit_b_16:getitem_5', 'mobilenet'], default='mobilenet+resnet18', help='image featurizer')
    args = parser.parse_args()
    main(args.src, args.dst, short=args.short, featurizer=args.featurizer, dedup_threshold=args.dedup)
