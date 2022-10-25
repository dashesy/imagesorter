# imagesorter
Sort images based on similarity features.
The default featurizer is 'mobilenet+resnet18' hybrid, which performs the best combining OD and Classification features.

# Usage

```bash
python sort.py /path/to/input/directory /path/to/output/directory/
```

To use different featurizer

```bash
python sort.py /path/to/input/directory /path/to/output/directory/ --featurizer mobilenet
```

# TODO

Paralelize and cache to disk to handle large number of images. (Currently fine up to few hundreds)

