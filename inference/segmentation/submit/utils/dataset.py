import torch
import os
import json
import copy
import cv2
import numpy as np
from skimage.io import imread
from glog import logger
import pathlib
import pandas as pd

__all__ = ["ImageNetSegmentationTest"]


def read_img(x: str):
    img = cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
    if img is None:
        logger.warning(f"Can not read image {x} with OpenCV, switching to scikit-image")
        img = imread(x)
    return img


def read_mask(x: str):
    mask = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning(f"Can not read image {x} with OpenCV, switching to scikit-image")
        mask = imread(x, as_gray=True)
    return np.expand_dims(mask, axis=-1)


class ImageNetSegmentationTest(torch.utils.data.Dataset):
    """ImageNet dataset for segmentation"""

    def __init__(self, images_path, transform=None):
        self.transform = transform
        self.images_dir = pathlib.Path(images_path)

        set_data = self.images_dir.stem

        self.images_list = pd.read_csv(
            os.path.join(self.images_dir, set_data + ".txt"), sep=" ", header=None
        )
        self.images_list = [[x[0], []] for x in self.images_list.values]

        with open(os.path.join(self.images_dir, "idx_to_name.json")) as fp:
            self.idx_to_name = json.load(fp)

        self.num_classes = len(self.idx_to_name) + 1  # plus background

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        relative_path, _ = self.images_list[idx]
        basename = relative_path.split(os.path.sep)[-1]

        image = read_img(os.path.join(self.images_dir, relative_path + ".JPEG"))
        orig_im = copy.deepcopy(image)

        if self.transform is not None:
            image, _ = self.transform(image, image)

        return image.permute(2, 0, 1), basename, orig_im
