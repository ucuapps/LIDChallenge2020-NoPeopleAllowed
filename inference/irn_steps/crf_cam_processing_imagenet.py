import numpy as np
import os
import yaml
import cv2
from torch import multiprocessing
from torch.utils.data import DataLoader
from inference.common.utils import split_dataset, crf_inference
from model_training.common.datasets import ImageNetMLC
from model_training.common.augmentations import get_transforms


def _work(process_id, config):
    transform = get_transforms(config["data"]["transform"])
    dataset = ImageNetMLC(config["data"]["path"], transform)
    subsets = split_dataset(dataset, config["num_workers"])
    dataloader = DataLoader(subsets[process_id], shuffle=False, pin_memory=False)

    for iteration, (X, y, name) in enumerate(dataloader):
        X, name = X[0].numpy().transpose((1, 2, 0)), name[0]
        if os.path.exists(os.path.join(config["data"]["output_path"], name + ".png")):
            continue

        cam_dict = np.load(
            os.path.join(config["data"]["cams_dir"], name + ".npy"), allow_pickle=True
        ).item()

        cams = cam_dict["high_res"]
        keys = np.pad(cam_dict["keys"] + 1, (1, 0), mode="constant")

        fg_conf_cam = np.pad(
            cams,
            ((1, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=config["conf_fg_threshold"],
        )
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        pred = crf_inference(X, fg_conf_cam, n_labels=keys.shape[0])
        fg_conf = keys[pred]

        bg_conf_cam = np.pad(
            cams,
            ((1, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=config["conf_bg_threshold"],
        )
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        pred = crf_inference(X, bg_conf_cam, n_labels=keys.shape[0])
        bg_conf = keys[pred]

        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255  # not confident area
        conf[(bg_conf + fg_conf) == 0] = 0  # confident background

        if config["rescale_output"]:  # rescale to original size
            p_orig = "/".join(config["data"]["path"].split("/")[:-1])
            orig = cv2.imread(p_orig + "/" + name + ".JPEG")
            conf = cv2.resize(
                conf.astype("float32"),
                (orig.shape[1], orig.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        cv2.imwrite(
            os.path.join(config["data"]["output_path"], name + ".png"),
            conf.astype(np.uint8),
        )

        if iteration % 500 == 0:
            print(
                f"Process: {process_id}, Iteration: {iteration}/{len(subsets[process_id])}"
            )


def run():
    with open("inference/irn_steps/config/crf_cam_processing_imagenet.yaml") as fp:
        config = yaml.full_load(fp)

    print("[ ", end="")
    multiprocessing.spawn(
        _work, nprocs=config["num_workers"], args=(config,), join=True
    )
    print("]")


if __name__ == "__main__":
    run()
