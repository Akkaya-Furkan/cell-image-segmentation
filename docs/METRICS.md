# Evaluation protocol

All scores use the original image grid and reference integer instance masks.
Background is zero; positive labels identify separate nuclei. Label values may
be noncontiguous. Labels and prediction shapes are checked before scoring.

## Foreground scores

Let A be the reference foreground and B the predicted foreground:

- IoU = |A ∩ B| / |A ∪ B|
- Dice = 2 |A ∩ B| / (|A| + |B|)

The report averages each score over images (macro average), so large images do
not dominate. An empty reference and empty prediction score one; exactly one
empty foreground scores zero.

## Instance scores

Compute the pairwise instance IoU matrix, including background when computing
object areas but excluding it from matching. At each threshold, find a
one-to-one matching that maximizes the number of eligible pairs; summed IoU
breaks ties. A pair is eligible when IoU **>=** threshold. Matched pairs are TP;
unmatched predictions are FP; unmatched reference objects are FN.

AP(t) = TP / (TP + FP + FN).

`ap50` is AP(0.50). `ap50_95` averages AP(t) for the threshold grid defined in
SCOPE.md. These are DSB-style instance detection scores. They do not use object
confidence ranks, PR integration, COCO area bins or maximum-detection limits;
they must not be compared directly to COCO AP. Both masks empty score one.
A split/merge can have perfect foreground Dice while its instance AP is lower.
The synthetic tests explicitly exercise this distinction.

## Selection and quality control

Training uses binary foreground masks, binary cross-entropy plus soft Dice,
and random horizontal/vertical flips. Each epoch is evaluated on validation
Dice after instance postprocessing at threshold 0.5. Retain the best epoch.
Then choose the baseline Otsu multiplier from [0.8, 1.0, 1.2] and the model
probability threshold from [0.35, 0.50, 0.65] using validation Dice. Other
parameters are fixed: Gaussian sigma 1 pixel, minimum object size 20 pixels,
watershed seed spacing 7 pixels; fill foreground holes. The model uses Adam
with learning rate 0.002 and batch size 4. Inference restores probabilities
to the original image dimensions before thresholding and watershed.

The test split is used only after all these selections. Overlay examples are
the worst, upper-middle and best test images ranked by U-Net Dice; filenames
state that selection rather than implying a random visual sample. Features
include all reference and predicted objects, including border-touching ones.
Area, perimeter and centroid coordinates are in pixels, not physical units.
Border-touching is a QC flag; objects are not removed from evaluation.
