import torch
import os
import json
import numpy as np

from .common import read_img, read_mask

__all__ = ['ImageNetMLC', 'ImageNetMLCVal', 'ImageNetSegmentation', 'ImageNetIAL', 'ImageNetHuman']


class ImageNetMLC(torch.utils.data.Dataset):
    """ImageNet dataset for multi-label classification"""

    def __init__(self, images_list_path, transform, return_size=False):
        self.transform = transform
        self.parent_dir = os.path.abspath(os.path.join(images_list_path, os.pardir))
        self.return_size = return_size

        with open(images_list_path, 'r') as fp:
            self.images_list = json.load(fp)

        with open(os.path.join(self.parent_dir, 'idx_to_name.json')) as fp:
            self.idx_to_name = json.load(fp)

        self.num_classes = len(self.idx_to_name)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        relative_path, labels = self.images_list[idx]

        basename = relative_path.split(os.path.sep)[-1]
        full_path = os.path.join(self.parent_dir, relative_path + '.JPEG')

        image = read_img(full_path)
        shape = torch.tensor(image.shape[:2])
        image = self.transform(image, None).permute(2, 0, 1)

        label_encoded = torch.zeros(self.num_classes, dtype=torch.float32)
        label_encoded[np.array(labels) - 1] = 1
        if self.return_size:
            return image, label_encoded, basename, shape
        return image, label_encoded, basename


class ImageNetHuman(torch.utils.data.Dataset):
    """ImageNet dataset for multi-label classification"""

    def __init__(self, images_list_path, transform, return_size=False):
        self.transform = transform
        self.parent_dir = os.path.abspath(os.path.join(images_list_path, os.pardir))
        self.return_size = return_size

        with open(images_list_path, 'r') as fp:
            self.images_list = json.load(fp)

        with open(os.path.join(self.parent_dir, 'idx_to_name.json')) as fp:
            self.idx_to_name = json.load(fp)

        self.num_classes = len(self.idx_to_name)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        relative_path, label = self.images_list[idx]

        basename = relative_path.split(os.path.sep)[-1]
        full_path = os.path.join(self.parent_dir, relative_path + '.JPEG')

        image = read_img(full_path)
        shape = torch.tensor(image.shape[:2])
        image = self.transform(image, None).permute(2, 0, 1)

        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        if self.return_size:
            return image, label, basename, shape
        return image, label, basename


class ImageNetMLCVal(torch.utils.data.Dataset):
    """ImageNet dataset for multi-label classification"""

    def __init__(self, images_list_path, transform, return_size=False):
        self.transform = transform
        self.parent_dir = os.path.abspath(os.path.join(images_list_path, os.pardir))
        self.return_size = return_size

        with open(images_list_path, 'r') as fp:
            self.images_list = list(map(lambda x: x.strip(), fp.readlines()))

        with open(os.path.join(self.parent_dir, 'idx_to_name.json')) as fp:
            self.idx_to_name = json.load(fp)

        self.num_classes = len(self.idx_to_name)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        basename = self.images_list[idx]

        full_path = os.path.join(self.parent_dir, basename + '.JPEG')

        image = read_img(full_path)
        shape = torch.tensor(image.shape[:2])
        image = self.transform(image, None).permute(2, 0, 1)

        if self.return_size:
            return image, torch.tensor([]), basename, shape
        return image, torch.tensor([]), basename


class ImageNetIAL(torch.utils.data.Dataset):
    """ImageNet dataset for intgral attention learning (segmentation with soft labels)"""
    MASK_STRIDE = 8

    def __init__(self, images_list_path, attention_path, transform, return_names=False, return_labels=False):
        self.transform = transform
        self.attention_path = attention_path
        self.return_names = return_names
        self.return_labels = return_labels
        self.parent_dir = os.path.abspath(os.path.join(images_list_path, os.pardir))

        with open(images_list_path, 'r') as fp:
            self.images_list = json.load(fp)

        with open(os.path.join(self.parent_dir, 'idx_to_name.json')) as fp:
            self.idx_to_name = json.load(fp)

        self.num_classes = len(self.idx_to_name)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        relative_path, labels = self.images_list[idx]
        basename = relative_path.split(os.path.sep)[-1]

        full_path = os.path.join(self.parent_dir, relative_path + '.JPEG')

        image = read_img(full_path)
        image = self.transform(image, None).permute(2, 0, 1)
        image_shape = image.shape[-1]

        mask = torch.zeros(self.num_classes, image_shape // self.MASK_STRIDE, image_shape // self.MASK_STRIDE)
        for cl in labels:
            activation = torch.FloatTensor(
                read_mask(os.path.join(self.attention_path, f'{basename}_{cl - 1}.png'))) / 255
            mask[cl - 1] = activation.squeeze()

        rv = [image, mask]
        if self.return_names:
            rv.append(basename)
        if self.return_labels:
            labels_one_hot = torch.zeros(self.num_classes)
            labels_one_hot[np.array(labels) - 1] = 1
            rv.append(labels_one_hot)
        return rv


class ImageNetSegmentation(torch.utils.data.Dataset):
    """ImageNet dataset for segmentation"""

    def __init__(self, images_list_path, masks_path, transform=None):
        self.transform = transform
        self.masks_path = masks_path
        self.parent_dir = os.path.abspath(os.path.join(images_list_path, os.pardir))

        with open(images_list_path, 'r') as fp:
            self.images_list = json.load(fp)

        with open(os.path.join(self.parent_dir, 'idx_to_name.json')) as fp:
            self.idx_to_name = json.load(fp)

        self.num_classes = len(self.idx_to_name) + 1  # plus background

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        relative_path, _ = self.images_list[idx]
        basename = relative_path.split(os.path.sep)[-1]

        image = read_img(os.path.join(self.parent_dir, relative_path + '.JPEG'))
        mask = read_mask(os.path.join(self.masks_path, basename) + '.png')

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image.permute(2, 0, 1), mask.permute(2, 0, 1).squeeze(0)
