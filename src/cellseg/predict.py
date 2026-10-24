"""Apply the delivered model to one TIFF/PNG image on CPU."""
import argparse
from pathlib import Path
import numpy as np
import tifffile
import torch
from skimage.io import imread
from .segmentation import TinyUNet, instances, probability


def predict(image, checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = TinyUNet()
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return instances(probability(model, image, saved['input_size']) > saved['threshold'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--checkpoint', type=Path, default=Path('results/tiny_unet.pt'))
    args = p.parse_args()
    torch.set_num_threads(4)
    image = imread(args.image)
    labels = predict(image, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(args.output, labels.astype(np.uint16), compression='zlib')
    print(f'Wrote {args.output}; instances={len(np.unique(labels[labels > 0]))}')
