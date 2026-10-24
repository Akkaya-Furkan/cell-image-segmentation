import numpy as np
import torch
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation, transform
from skimage.feature import peak_local_max
from torch import nn


def normalize(image):
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 3:
        x = x[..., :3].mean(axis=-1)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite grayscale or RGB image')
    low, high = np.percentile(x, [1, 99.8])
    return np.clip((x - low) / max(float(high - low), 1e-6), 0, 1).astype(np.float32)


def instances(binary, min_size=20, min_distance=7):
    binary = morphology.remove_small_objects(np.asarray(binary, dtype=bool), min_size=min_size)
    binary = ndi.binary_fill_holes(binary)
    distance = ndi.distance_transform_edt(binary)
    peaks = peak_local_max(distance, min_distance=min_distance, labels=binary, exclude_border=False)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (y, x) in enumerate(peaks, 1):
        markers[y, x] = i
    # Ensure every connected foreground component receives at least one seed.
    components = measure.label(binary)
    next_id = len(peaks) + 1
    for region in measure.regionprops(components):
        coords = region.coords
        if not markers[coords[:, 0], coords[:, 1]].any():
            y, x = coords[np.argmax(distance[coords[:, 0], coords[:, 1]])]
            markers[y, x] = next_id
            next_id += 1
    return segmentation.watershed(-distance, markers, mask=binary).astype(np.int32)


def baseline(image, multiplier=1.):
    x = filters.gaussian(normalize(image), sigma=1., preserve_range=True)
    if x.max() == x.min():
        return np.zeros(x.shape, dtype=np.int32)
    return instances(x > filters.threshold_otsu(x) * multiplier)


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = block(1, 8)
        self.enc2 = block(8, 16)
        self.bottom = block(16, 32)
        self.dec2 = block(48, 16)
        self.dec1 = block(24, 8)
        self.out = nn.Conv2d(8, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.bottom(self.pool(b))
        c = nn.functional.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False)
        d = self.dec2(torch.cat([c, b], 1))
        d = nn.functional.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False)
        return self.out(self.dec1(torch.cat([d, a], 1)))


def model_input(image, size=256):
    return transform.resize(normalize(image), (size, size), preserve_range=True, anti_aliasing=True).astype(np.float32)


def probability(model, image, size=256):
    with torch.inference_mode():
        x = torch.from_numpy(model_input(image, size))[None, None]
        p = model(x).sigmoid()[0, 0].numpy()
    return transform.resize(p, image.shape[:2], preserve_range=True, anti_aliasing=False)
