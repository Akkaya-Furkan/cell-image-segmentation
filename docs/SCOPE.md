# One-day scope and acceptance criteria

## Question
Can nuclei be segmented reliably in a small public microscopy dataset?
Here, “cells” means fluorescently imaged nuclei, not whole-cell boundaries.

## Frozen plan (before implementation and evaluation)
Use the public DSB2018 image/instance-mask archive distributed by StarDist:
https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip
Dataset origin and CC0 terms: https://bbbc.broadinstitute.org/BBBC038.
Select lexicographically first 32 paired images from the archive's training
partition: first 24 for training, next 8 for validation. Select first 12
paired images from its test partition for the held-out evaluation. Publish
IDs and SHA-256 hashes. These are design targets, not measured results.

Compare Gaussian smoothing + Otsu threshold + distance-transform watershed
against a small randomly initialized U-Net trained on CPU. Resize only model
inputs to 256 × 256; score instance masks at the original resolution. Select
thresholds using validation images only. Fix seed to 25. No test-based tuning.

## Acceptance (targets)
- Exactly 24 train, 8 validation and 12 test images, with disjoint IDs and hashes.
- Both methods evaluated on all 12 test reference instance masks.
- Mean per-image foreground Dice at least 0.70 for at least one method.
- Report foreground IoU/Dice, instance AP50 and mean AP at IoU thresholds
  0.50 through 0.95 in increments of 0.05. AP is the DSB-style instance score
  TP/(TP+FP+FN) after one-to-one matching, not a confidence-ranked PR integral.
- Produce 24 test prediction masks, 3 example overlay figures, per-image
  metrics, per-object morphology features and a machine-readable summary.
- Synthetic metric/postprocessing tests and an end-to-end training/inference
  smoke test pass with pytest; GitHub Actions runs these without network data.
- Provide a conda environment, download script, reproducible CLI, trained
  checkpoint, English README with measured results and limitations.
- Commit and push main, verify the remote hash, and read the CI result.

## Exclusions
Whole-cell segmentation, clinical use, GPU benchmarking, exhaustive model
search, large-scale training and claims of external biological generalization
are not done. A small held-out subset cannot establish universal reliability.
Performance targets may fail; report failure without changing this scope.
