"""Run the frozen CPU experiment: python -m cellseg.benchmark."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import resource
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from skimage import measure, segmentation, transform
from torch import nn

from .metrics import evaluate
from .segmentation import TinyUNet, baseline, instances, model_input, normalize, probability


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load_data(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    groups = {s: [] for s in ['train', 'validation', 'test']}
    for record in manifest['records']:
        for kind in ['image', 'mask']:
            path = root / record[kind]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record[kind + '_sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
        image, mask = [tifffile.imread(root / record[k]) for k in ['image', 'mask']]
        if image.shape[:2] != mask.shape or mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            raise ValueError('Invalid image/mask pair')
        groups[record['split']].append((record['id'], image, mask))
    expected = {'train': 24, 'validation': 8, 'test': 12}
    if {s: len(v) for s, v in groups.items()} != expected:
        raise ValueError('Frozen split counts violated')
    for key in ['id', 'image_sha256', 'mask_sha256']:
        if len({r[key] for r in manifest['records']}) != sum(expected.values()):
            raise ValueError(f'Duplicate {key}')
    return manifest, groups


def train(model, training, validation, epochs=40, size=256):
    x = torch.from_numpy(np.stack([model_input(img, size) for _, img, _ in training]))[:, None]
    y = torch.from_numpy(np.stack([transform.resize(mask > 0, (size, size), order=0,
                            preserve_range=True, anti_aliasing=False) for _, _, mask in training]).astype(np.float32))[:, None]
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    history, best, best_state = [], -1., None
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x))
        losses = []
        for indices in order.split(4):
            xb, yb = x[indices], y[indices]
            if random.random() < .5:
                xb, yb = xb.flip(-1), yb.flip(-1)
            if random.random() < .5:
                xb, yb = xb.flip(-2), yb.flip(-2)
            logits = model(xb)
            p = logits.sigmoid()
            soft_dice = (2 * (p * yb).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + yb.sum((1, 2, 3)) + 1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb) + 1 - soft_dice.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        score = float(np.mean([evaluate(mask, instances(probability(model, img, size) > .5))['dice']
                               for _, img, mask in validation]))
        history.append({'epoch': epoch, 'training_loss': float(np.mean(losses)), 'validation_dice': score})
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_state)
    return history


def run(root, output, epochs=40, threads=4):
    started = time.perf_counter()
    random.seed(25)
    np.random.seed(25)
    torch.manual_seed(25)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    manifest, groups = load_data(root)
    output.mkdir(parents=True, exist_ok=True)
    masks_dir, figures_dir = output / 'masks', output.parent / 'figures'
    masks_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    model = TinyUNet()
    train_started = time.perf_counter()
    history = train(model, groups['train'], groups['validation'], epochs)
    training_seconds = time.perf_counter() - train_started
    write_csv(output / 'training_history.csv', history)
    # Only validation labels participate in hyperparameter selection.
    val_probs = {id_: probability(model, img) for id_, img, _ in groups['validation']}
    selection = []
    for method, candidates in [('watershed', [.8, 1., 1.2]), ('unet', [.35, .5, .65])]:
        for candidate in candidates:
            score = np.mean([evaluate(mask, baseline(img, candidate) if method == 'watershed'
                            else instances(val_probs[id_] > candidate))['dice'] for id_, img, mask in groups['validation']])
            selection.append({'method': method, 'threshold': candidate, 'validation_dice': float(score)})
    chosen = {method: max([r for r in selection if r['method'] == method], key=lambda r: r['validation_dice'])['threshold']
              for method in ['watershed', 'unet']}
    write_csv(output / 'validation_selection.csv', selection)
    checkpoint = {'state_dict': model.state_dict(), 'threshold': chosen['unet'], 'input_size': 256, 'seed': 25}
    torch.save(checkpoint, output / 'tiny_unet.pt')
    # Reload the saved artifact so reported inference exercises the delivered checkpoint.
    saved = torch.load(output / 'tiny_unet.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    rows, features, visuals = [], [], []
    for id_, image, truth in groups['test']:
        predictions = {'watershed': baseline(image, chosen['watershed']),
                       'unet': instances(probability(model, image, saved['input_size']) > saved['threshold'])}
        for method, pred in predictions.items():
            metrics = evaluate(truth, pred)
            rows.append({'id': id_, 'method': method, **metrics})
            tifffile.imwrite(masks_dir / f'{id_}_{method}.tif', pred.astype(np.uint16), compression='zlib')
        for source, labels in [('reference', truth), *predictions.items()]:
            for region in measure.regionprops(labels):
                features.append({'id': id_, 'source': source, 'label': region.label, 'area_px': region.area,
                                 'centroid_y_px': region.centroid[0], 'centroid_x_px': region.centroid[1],
                                 'eccentricity': region.eccentricity, 'perimeter_px': region.perimeter,
                                 'border_touching': bool(region.bbox[0] == 0 or region.bbox[1] == 0 or
                                                         region.bbox[2] == labels.shape[0] or region.bbox[3] == labels.shape[1])})
        visuals.append((id_, image, truth, predictions, rows[-1]['dice']))
    write_csv(output / 'per_image.csv', rows)
    write_csv(output / 'features.csv', features)
    ranked = sorted(visuals, key=lambda v: v[-1])
    for label, example in [('worst', ranked[0]), ('median', ranked[len(ranked) // 2]), ('best', ranked[-1])]:
        id_, image, truth, preds, _ = example
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for ax, (title, mask) in zip(axes, [('Input', None), ('Reference', truth), *preds.items()]):
            x = normalize(image)
            ax.imshow(x, cmap='gray', vmin=0, vmax=1)
            if mask is not None:
                ax.imshow(segmentation.mark_boundaries(x, mask, color=(1, .3, .1)))
                if title in preds:
                    score = evaluate(truth, mask)
                    title += f"\nDice {score['dice']:.3f}; AP50 {score['ap50']:.3f}"
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'{label.capitalize()} U-Net Dice example | {id_[:16]}')
        fig.tight_layout(rect=(0, 0, 1, .90))
        fig.savefig(figures_dir / f'overlay_{label}.png', dpi=140)
        plt.close(fig)
    summary = {'seed': 25, 'device': 'cpu', 'threads': threads, 'epochs': epochs,
               'model_parameters': sum(p.numel() for p in model.parameters()),
               'counts': manifest['counts'], 'unique_image_hashes': len({r['image_sha256'] for r in manifest['records']}),
               'unique_mask_hashes': len({r['mask_sha256'] for r in manifest['records']}),
               'chosen_thresholds': chosen, 'best_epoch': max(history, key=lambda r: r['validation_dice'])['epoch'],
               'methods': {m: {metric: float(np.mean([r[metric] for r in rows if r['method'] == m]))
                                   for metric in ['iou', 'dice', 'ap50', 'ap50_95']} for m in chosen},
               'test_reference_instances': sum(r['true_instances'] for r in rows if r['method'] == 'unet'),
               'prediction_masks': len(list(masks_dir.glob('*.tif'))), 'feature_rows': len(features),
               'overlay_figures': len(list(figures_dir.glob('overlay_*.png'))),
               'training_seconds': training_seconds, 'total_seconds': time.perf_counter() - started,
               'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               'platform': platform.platform(), 'python': platform.python_version(),
               'versions': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'scipy', 'scikit-image', 'tifffile', 'matplotlib']}}
    summary['acceptance'] = {'split_counts': summary['counts'] == {'train': 24, 'validation': 8, 'test': 12},
                             'evaluation_rows': len(rows) == 24,
                             'dice_target': max(m['dice'] for m in summary['methods'].values()) >= .70,
                             'prediction_masks': summary['prediction_masks'] == 24,
                             'overlays': summary['overlay_figures'] == 3}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('results'))
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.epochs < 1 or args.threads < 1:
        p.error('epochs and threads must be positive')
    run(args.data, args.output, args.epochs, args.threads)
