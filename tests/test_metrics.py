import numpy as np
import pytest
from cellseg.metrics import evaluate


def test_perfect_noncontiguous_and_relabelled():
    gt = np.array([[0, 9, 9], [0, 0, 0], [37, 37, 0]])
    pred = np.array([[0, 2, 2], [0, 0, 0], [5, 5, 0]])
    result = evaluate(gt, pred)
    for metric in ['dice', 'iou', 'ap50', 'ap50_95']:
        assert result[metric] == 1
    assert result['true_instances'] == 2


def test_merge_is_not_perfect_instance_segmentation():
    gt = np.array([[1, 1, 2, 2]])
    pred = np.ones_like(gt)
    result = evaluate(gt, pred)
    assert result['dice'] == result['iou'] == 1
    assert result['ap50'] == .5
    assert result['ap50_95'] == pytest.approx(.05)


def test_split_is_penalized():
    result = evaluate(np.ones((1, 4), dtype=int), np.array([[1, 1, 2, 2]]))
    assert result['ap50'] == .5
    assert result['ap50_95'] == pytest.approx(.05)


def test_hand_calculated_partial_overlap():
    result = evaluate(np.array([[1, 1, 0]]), np.array([[0, 1, 1]]))
    assert result['dice'] == .5
    assert result['iou'] == pytest.approx(1 / 3)
    assert result['ap50'] == 0


@pytest.mark.parametrize('which', ['both', 'truth', 'prediction'])
def test_empty_masks(which):
    a, b = np.zeros((4, 4), dtype=int), np.zeros((4, 4), dtype=int)
    if which == 'truth':
        a[1:3, 1:3] = 1
    elif which == 'prediction':
        b[1:3, 1:3] = 1
    result = evaluate(a, b)
    assert result['dice'] == result['ap50_95'] == (1 if which == 'both' else 0)


@pytest.mark.parametrize('truth,pred', [
    (np.zeros((2, 2), int), np.zeros((3, 3), int)),
    (np.array([[-1]]), np.array([[0]])),
    (np.array([[.1]]), np.array([[0]]))])
def test_invalid_masks(truth, pred):
    with pytest.raises(ValueError):
        evaluate(truth, pred)
