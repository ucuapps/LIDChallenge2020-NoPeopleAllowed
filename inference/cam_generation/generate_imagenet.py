import yaml
import torch
import os

from inference.cam_generation.extractor_grub_cut import ActivationExtractor
from model_training.common.datasets import ImageNetMLCVal
from model_training.common.augmentations import get_transforms

with open(os.path.join(os.path.dirname(__file__), 'config', 'imagenet.yaml')) as config_file:
    config = yaml.full_load(config_file)

train_transform = get_transforms(config['train']['transform'])
val_transform = get_transforms(config['val']['transform'])

# train_ds = ImageNetMLC(config['train']['input_path'], transform=train_transform, return_size=True)
val_ds = ImageNetMLCVal(config['val']['input_path'], transform=val_transform, return_size=True)

# train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=12)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'], shuffle=True, num_workers=12)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device {device}')

extractor = ActivationExtractor(config, None, val_dl, device)
extractor.extract()
