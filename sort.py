import os
import os.path as op
import argparse
import shutil
import imagesorter.sorter as sorter

def main(srcdir: str, dstdir: str):
    sorted = sorter.sort(srcdir)
    if not sorted:
        return
    os.makedirs(dstdir, exist_ok=True)
    length = len(str(len(sorted)))
    max_src_length = max([len(fname) for fname in [op.basename(fname) for fname in sorted]])
    with open(op.join(dstdir, "sorted.csv"), "w") as wfp:
        for ii in range(len(sorted)):
            src = sorted[ii]
            ext = op.basename(src).rjust(max_src_length, "-")
            dst = str(ii).rjust(length, '0') + "-"+ ext
            print(f"{src},{dst}", file=wfp)
            shutil.copy(src, op.join(dstdir, dst))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sort input images into output directory with numerical names')
    parser.add_argument('src', help='path to input directory of images to be sorted')
    parser.add_argument('dst', help='path to output directory to add sorted images')
    args = parser.parse_args()
    main(args.src, args.dst)
