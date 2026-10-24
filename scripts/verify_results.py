#!/usr/bin/env python3
"""Re-read delivered artifacts, recompute scores, and replay saved-model inference."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
import torch
from skimage.io import imread

from cellseg.benchmark import load_data
from cellseg.metrics import evaluate
from cellseg.predict import predict
from cellseg.segmentation import baseline


def verify():
    root, output = Path('data'), Path('results')
    torch.set_num_threads(4)
    manifest, groups = load_data(root)
    summary = json.loads((output / 'summary.json').read_text())
    assert json.loads((output / 'data_manifest.json').read_text()) == manifest
    with (output / 'per_image.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 24
    scores = {(r['id'], r['method']): r for r in rows}
    assert len(scores) == 24
    expected_masks, replayed = set(), 0
    for id_, image, truth in groups['test']:
        for method in ['watershed', 'unet']:
            path = output / 'masks' / f'{id_}_{method}.tif'
            expected_masks.add(path.name)
            mask = tifffile.imread(path)
            for key, value in evaluate(truth, mask).items():
                assert np.isclose(value, float(scores[id_, method][key]), atol=1e-12)
            regenerated = (predict(image, output / 'tiny_unet.pt') if method == 'unet' else
                           baseline(image, summary['chosen_thresholds']['watershed']))
            np.testing.assert_array_equal(mask, regenerated)
            replayed += 1
    assert {p.name for p in (output / 'masks').glob('*.tif')} == expected_masks
    for method, values in summary['methods'].items():
        for metric, value in values.items():
            assert np.isclose(value, np.mean([float(r[metric]) for r in rows if r['method'] == method]))
    with (output / 'features.csv').open() as f:
        features = list(csv.DictReader(f))
    assert len(features) == summary['feature_rows']
    for id_, _, truth in groups['test']:
        for source in ['reference', 'watershed', 'unet']:
            mask = truth if source == 'reference' else tifffile.imread(output / 'masks' / f'{id_}_{source}.tif')
            selected = [r for r in features if r['id'] == id_ and r['source'] == source]
            assert len(selected) == len(np.unique(mask[mask > 0]))
            for row in selected:
                assert int(row['area_px'].split('.')[0]) == np.count_nonzero(mask == int(row['label']))
    for label in ['best', 'median', 'worst']:
        assert imread(Path('figures') / f'overlay_{label}.png').ndim == 3
    suites = ET.parse(output / 'local-tests.xml').getroot().findall('testsuite')
    tests = sum(int(s.attrib['tests']) for s in suites)
    failures = sum(int(s.attrib['failures']) + int(s.attrib['errors']) for s in suites)
    assert tests > 0 and failures == 0
    artifacts = sorted([p for p in output.rglob('*') if p.is_file() and p.name != 'verification.json'] + list(Path('figures').glob('*.png')))
    report = {'expected_split_counts': {'train': 24, 'validation': 8, 'test': 12},
              'actual_split_counts': {s: len(g) for s, g in groups.items()},
              'expected_metric_rows': 24, 'actual_metric_rows': len(rows),
              'expected_prediction_masks': 24, 'actual_prediction_masks': len(expected_masks),
              'replayed_exact_masks': replayed, 'expected_overlays': 3, 'actual_overlays': len(list(Path('figures').glob('overlay_*.png'))),
              'feature_rows': len(features), 'pytest_tests': tests, 'pytest_failures_and_errors': failures,
              'pytest_seconds': sum(float(s.attrib['time']) for s in suites),
              'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'sha256'}, indent=2))


if __name__ == '__main__':
    verify()
