import numpy as np
import tifffile
import torch
from skimage.draw import disk
from cellseg.benchmark import train
from cellseg.metrics import evaluate
from cellseg.predict import predict
from cellseg.segmentation import TinyUNet, baseline, instances, normalize


def synthetic():
    image = np.zeros((64, 64), np.float32)
    labels = np.zeros((64, 64), np.int32)
    for i, center in enumerate([(18, 18), (44, 44)], 1):
        rr, cc = disk(center, 8, shape=image.shape)
        image[rr, cc], labels[rr, cc] = 1, i
    return image, labels


def test_baseline_two_disks():
    image, labels = synthetic()
    score = evaluate(labels, baseline(image))
    assert score['dice'] > .95
    assert score['ap50'] == 1


def test_empty_and_border_postprocessing():
    assert not instances(np.zeros((16, 16))).any()
    binary = np.zeros((32, 32), bool)
    binary[:8, :8] = True
    labels = instances(binary)
    assert labels.max() == 1
    assert np.array_equal(labels > 0, binary)
    assert np.isfinite(normalize(np.ones((8, 8)))).all()
    assert not baseline(np.ones((8, 8))).any()


def test_training_checkpoint_native_size_inference(tmp_path):
    torch.manual_seed(25)
    torch.set_num_threads(1)
    image, labels = synthetic()
    model = TinyUNet()
    before = model.out.weight.detach().clone()
    history = train(model, [('train', image, labels)], [('val', image, labels)], epochs=1, size=32)
    assert len(history) == 1
    assert np.isfinite(history[0]['training_loss'])
    assert not torch.equal(before, model.out.weight)
    checkpoint = tmp_path / 'model.pt'
    torch.save({'state_dict': model.state_dict(), 'input_size': 32, 'threshold': .5}, checkpoint)
    result = predict(image, checkpoint)
    assert result.shape == labels.shape
    assert np.issubdtype(result.dtype, np.integer)
    path = tmp_path / 'prediction.tif'
    tifffile.imwrite(path, result)
    np.testing.assert_array_equal(tifffile.imread(path), result)
