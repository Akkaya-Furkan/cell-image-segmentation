#!/usr/bin/env python3
"""Download a checksum-pinned public archive; extract only the frozen subset."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip'
SHA256 = 'e44921950edce378063aa4457e625581ba35b4c2dbd9a07c19d48900129f386f'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, archive=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cached = root / 'dsb2018.zip'
    if archive is not None:
        if Path(archive).resolve() != cached.resolve():
            shutil.copyfile(archive, cached)
    if not cached.exists():
        partial = cached.with_suffix('.part')
        try:
            urllib.request.urlretrieve(URL, partial)
            if digest(partial) != SHA256:
                raise ValueError('Archive checksum mismatch')
            partial.replace(cached)
        finally:
            partial.unlink(missing_ok=True)
    if digest(cached) != SHA256:
        raise ValueError('Archive checksum mismatch; delete cached archive and retry')
    records = []
    with zipfile.ZipFile(cached) as z:
        for partition, count in [('train', 32), ('test', 12)]:
            prefix = f'dsb2018/{partition}/images/'
            images = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith('.tif'))[:count]
            if len(images) != count:
                raise ValueError('Insufficient paired images')
            for i, image in enumerate(images):
                split = ('train' if i < 24 else 'validation') if partition == 'train' else 'test'
                record = {'id': Path(image).stem, 'split': split, 'archive_partition': partition}
                for kind, member in [('image', image), ('mask', image.replace('/images/', '/masks/'))]:
                    payload = z.read(member)
                    dest = root / split / (kind + 's') / Path(member).name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(payload)
                    record[kind] = dest.relative_to(root).as_posix()
                    record[kind + '_sha256'] = hashlib.sha256(payload).hexdigest()
                records.append(record)
    for key in ['id', 'image_sha256', 'mask_sha256']:
        if len({r[key] for r in records}) != len(records):
            raise ValueError(f'Duplicate {key} across selected records')
    manifest = {'url': URL, 'archive_sha256': SHA256, 'archive_bytes': cached.stat().st_size,
                'counts': {s: sum(r['split'] == s for r in records) for s in ['train', 'validation', 'test']},
                'records': records}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data')
    parser.add_argument('--archive', help='Optional already downloaded archive; checksum still verified')
    args = parser.parse_args()
    prepare(args.root, args.archive)
