import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from torch.utils.data import Subset


def split_dataset(dataset, n_splits):
    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def crf_inference(img, labels, t=10, n_labels=21, gt_prob=0.7):
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)
