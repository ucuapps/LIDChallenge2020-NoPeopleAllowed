import numpy as np
import os
import yaml
import torch
from torch import multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader
from inference.cam_generation.grad_cam import get_cam_grad_extractor
from inference.common.utils import split_dataset
from model_training.common.datasets import ImageNetMLC
from model_training.common.augmentations import get_transforms


def _work(process_id, config):
    def __horizontal_flip_tta(x, y, tta=True):
        maps, y_predicted = extractor.forward(x, y)
        if tta:
            maps_flipped, _ = extractor.forward(x.flip(-1), y)
            return maps + maps_flipped.flip(-1), y_predicted
        else:
            return maps, y_predicted

    device = config["devices"][process_id]
    scales = config["scales"]
    full_size = config["data"]["transform"]["size"]
    small_size = full_size // 4

    extractor = get_cam_grad_extractor(config, device)

    transform = get_transforms(config["data"]["transform"])
    dataset = ImageNetMLC(config["data"]["path"], transform)
    subsets = split_dataset(dataset, len(config["devices"]))
    dataloader = DataLoader(subsets[process_id], shuffle=False, pin_memory=False)

    for iteration, (X, y, name) in enumerate(dataloader):
        X, y, name = (
            X.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
            name[0],
        )

        if os.path.exists(os.path.join(config["data"]["output_path"], name + ".npy")):
            continue

        if y.sum() == 0:
            y = None
            _, y_pred = __horizontal_flip_tta(
                x=F.interpolate(
                    X, scale_factor=1.0, mode="bilinear", align_corners=False
                ),
                y=y,
                tta=False,
            )
            label_encoded = torch.zeros(config["model"]["classes"], dtype=torch.float32)
            label_encoded[y_pred.data.cpu().numpy()] = 1
            y = label_encoded.unsqueeze_(0)

        outputs = [
            __horizontal_flip_tta(
                x=F.interpolate(
                    X, scale_factor=scale, mode="bilinear", align_corners=False
                ),
                y=y,
            )[0]
            for scale in scales
        ]

        highres_cam = torch.sum(
            torch.stack(
                [
                    F.interpolate(
                        output[None],
                        size=(full_size, full_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    for output in outputs
                ],
                dim=0,
            ),
            dim=0,
        )
        lowres_cam = torch.sum(
            torch.stack(
                [
                    F.interpolate(
                        output[None],
                        size=(small_size, small_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    for output in outputs
                ],
                dim=0,
            ),
            dim=0,
        )
        highres_cam = highres_cam.relu_() / highres_cam.max()
        lowres_cam = lowres_cam.relu_() / lowres_cam.max()

        np.save(
            os.path.join(config["data"]["output_path"], name + ".npy"),
            {
                "keys": torch.where(y)[1].cpu().numpy(),
                "cam": lowres_cam[0].cpu().numpy(),
                "high_res": highres_cam[0].cpu().numpy(),
            },
        )

        if iteration % 100 == 0:
            print(
                f"Device: {process_id}, Iteration: {iteration}/{len(subsets[process_id])}"
            )


def run():
    with open(os.path.join(os.path.dirname(__file__), 'config', "multiscale_imagenet_cam.yaml")) as fp:
        config = yaml.full_load(fp)

    print("[ ", end="")
    multiprocessing.spawn(
        _work, nprocs=len(config["devices"]), args=(config,), join=True
    )
    print("]")


if __name__ == "__main__":
    run()
