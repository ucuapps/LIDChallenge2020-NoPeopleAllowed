from .base import *
from .classification import *
from .segmentation import *
from .irn import *


def get_model_adapter(config, log_path):
    if config["task"] == "segmentation":
        return SegmentationModelAdapter(config, log_path)
    elif config["task"] == "cam_generation":
        return ClassificationModelAdapter(config, log_path)
    elif config["task"] == "irn_train":
        return IRNModelAdapter(config, log_path)
    else:
        raise ValueError(f'Unrecognized task [{config["task"]}]')
