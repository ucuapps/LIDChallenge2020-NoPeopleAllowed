import torch
import os
import json
import numpy as np
import cv2

from model_training.common.datasets.common import read_img, read_mask
from model_training.common.datasets.irn.indexing import PathIndex

__all__ = ['ImageNetAffinityDataset']


class ImageNetAffinityDataset(torch.utils.data.Dataset):
    def __init__(self, images_list_path, transform, mask_dir, image_output_size=512):
        self.transform = transform
        self.parent_dir = os.path.abspath(os.path.join(images_list_path, os.pardir))
        self.mask_dir = mask_dir

        path_index = PathIndex(radius=10, default_size=(image_output_size // 4, image_output_size // 4))
        self.extract_aff_lab_func = GetAffinityLabelFromIndices(path_index.src_indices, path_index.dst_indices)

        with open(images_list_path, 'r') as fp:
            self.images_list = json.load(fp)

        with open(os.path.join(self.parent_dir, 'idx_to_name.json')) as fp:
            self.idx_to_name = json.load(fp)

        self.num_classes = len(self.idx_to_name)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        relative_path, _ = self.images_list[idx]

        basename = relative_path.split(os.path.sep)[-1]
        full_path = os.path.join(self.parent_dir, relative_path + '.JPEG')

        image = read_img(full_path)
        mask = read_mask(os.path.join(self.mask_dir, basename + '.png'))

        image, mask = self.transform(image, mask)

        out = {
            'image': image.permute(2, 0, 1),
            'name': basename
        }
        reduced_mask = cv2.resize(mask, (mask.shape[0] // 4, mask.shape[1] // 4), interpolation=cv2.INTER_NEAREST)

        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = \
            self.extract_aff_lab_func(reduced_mask)

        return out


class GetAffinityLabelFromIndices(object):

    def __init__(self, indices_from, indices_to):
        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):
        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 201), np.less(segm_label_to, 201))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)


if __name__ == '__main__':
    from model_training.common.augmentations import get_transforms

    transform_config = {
        'size': 512,
        'augmentation_scope': 'strong',
        'images_normalization': 'default',
        'images_output_format_type': 'float',
        'size_transform': 'resize',
        'masks_normalization': 'none',
        'masks_output_format_type': 'numpy'
    }

    transform = get_transforms(transform_config)

    ds = ImageNetAffinityDataset(
        images_list_path='/datasets/LID/ILSVRC/Data/DET/train/train_balanced.json',
        transform=transform,
        mask_dir='/datasets/LID/ILSVRC/Data/DET/train/outputs/irn_label',
        image_output_size=512
    )
    pack = ds[0]
    print(pack['image'].shape)
    print(pack['name'])
    print(len(pack['aff_bg_pos_label']))
