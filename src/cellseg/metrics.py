"""Foreground overlap and DSB-style one-to-one instance AP (not PR-AUC)."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError('Expected same-shape 2D instance masks')
    if np.any(truth < 0) or np.any(prediction < 0):
        raise ValueError('Instance labels must be nonnegative')
    if not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError('Instance labels must be integers')
    a, b = truth > 0, prediction > 0
    inter, union = np.count_nonzero(a & b), np.count_nonzero(a | b)
    size = np.count_nonzero(a) + np.count_nonzero(b)
    # Always retain the zero/background row, even for full-foreground masks.
    tids, pids = np.union1d([0], np.unique(truth)), np.union1d([0], np.unique(prediction))
    t, p = np.searchsorted(tids, truth), np.searchsorted(pids, prediction)
    joint = np.bincount((t * len(pids) + p).ravel(), minlength=len(tids) * len(pids)).reshape(len(tids), len(pids))
    overlaps = joint[1:, 1:]
    denom = joint.sum(1)[1:, None] + joint.sum(0)[None, 1:] - overlaps
    iou = np.divide(overlaps, denom, out=np.zeros_like(overlaps, dtype=float), where=denom > 0)
    scores = []
    for threshold in np.linspace(.5, .95, 10):
        # Cardinality is primary, summed IoU only breaks ties.
        eligible = iou >= threshold
        rows, cols = linear_sum_assignment(-(eligible.astype(float) + iou / (1 + min(iou.shape))))
        tp = int(eligible[rows, cols].sum())
        denominator = len(tids) - 1 + len(pids) - 1 - tp
        scores.append(tp / denominator if denominator else 1.)
    return {'iou': inter / union if union else 1., 'dice': 2 * inter / size if size else 1.,
            'ap50': scores[0], 'ap50_95': float(np.mean(scores)),
            'true_instances': len(tids) - 1, 'pred_instances': len(pids) - 1}
