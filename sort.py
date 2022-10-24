import os
import os.path as op
import argparse
import shutil
import imagesorter.sorter as sorter

def main(srcdir: str, dstdir: str, short=False, featurizer='resnet18'):
    sorted = sorter.sort(srcdir, featurizer=featurizer)
    if not sorted:
        return
    os.makedirs(dstdir, exist_ok=True)
    length = len(str(len(sorted)))
    max_src_length = max([len(fname) for fname in [op.basename(fname) for fname in sorted]])
    with open(op.join(dstdir, "sorted.csv"), "w") as wfp:
        for ii in range(len(sorted)):
            src = sorted[ii]
            if short:
                ext = op.splitext(src)[1]
            else:
                ext = "-" + op.basename(src).rjust(max_src_length, "-")
            dst = str(ii).rjust(length, '0') + ext
            print(f"{src},{dst}", file=wfp)
            shutil.copy(src, op.join(dstdir, dst))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sort input images into output directory with numerical names')
    parser.add_argument('src', help='path to input directory of images to be sorted')
    parser.add_argument('dst', help='path to output directory to add sorted images')
    parser.add_argument('-s', '--short', action='store_true', help='only keep the short numerical names')
    parser.add_argument('-f', '--featurizer', choices=['resnet18', 'vit_b_16', 'vit_b_16:getitem_5', 'mobilenet', 'mobilenet_resnet18'], default='resnet18', help='image featurizer')
    args = parser.parse_args()
    main(args.src, args.dst, short=args.short, featurizer=args.featurizer)
