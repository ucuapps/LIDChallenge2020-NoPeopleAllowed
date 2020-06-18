import yaml
import numpy as np
import os
import torch
from torch import multiprocessing
import torch.nn.functional as F

from torch.utils.data import DataLoader
from model_training.common.models import get_network
from inference.common.utils import split_dataset
from model_training.common.datasets import ImageNetMLC
from model_training.common.augmentations import get_transforms
from model_training.common.datasets.irn import indexing


def _work(process_id, config):
    device = config["devices"][process_id]

    model = get_network(config['model'])
    model.load_state_dict(torch.load(config['model']['weights_path'], map_location=device)['model'])
    model.to(device)
    model.eval()

    transform = get_transforms(config["data"]["transform"])
    dataset = ImageNetMLC(config["data"]["path"], transform, return_size=True)
    subsets = split_dataset(dataset, len(config["devices"]))
    dataloader = DataLoader(subsets[process_id], shuffle=False, pin_memory=False)

    with torch.no_grad():
        for iteration, (X, y, name, orig_size) in enumerate(dataloader):
            X, y, name, orig_size = (
                X.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
                name[0],
                orig_size[0]
            )

            if os.path.exists(os.path.join(config["data"]["output_path"], name + ".npy")):
                continue

            X_tta = torch.cat([X, X.flip(-1)], dim=0)
            edge, _ = model(X_tta)
            edge = torch.sigmoid(edge[0] / 2 + edge[1].flip(-1) / 2)

            cam_dict = np.load(config['data']['cam_path'] + '/' + name + '.npy', allow_pickle=True).item()
            cams = torch.from_numpy(cam_dict['cam']).to(device)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cams = F.interpolate(cams.unsqueeze(1), size=edge.shape[1:],
                                 mode='bilinear', align_corners=False).squeeze(1)
            rw = indexing.propagate_to_edge(cams, edge, beta=config['beta'], exp_times=config['exp_times'], radius=5,
                                            device=device)

            rw_up = F.interpolate(rw, size=(orig_size[0], orig_size[1]), mode='bilinear', align_corners=False)
            rw_up = rw_up.relu_() / torch.max(rw_up)
            np.save(
                os.path.join(config["data"]["output_path"], name + ".npy"),
                {
                    "keys": keys,
                    "map": rw_up.squeeze(0).cpu().numpy()
                },
            )

            # rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=config['sem_seg_bg_thres'])[0]
            # rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()
            # rw_pred = keys[rw_pred]

            if iteration % 100 == 0:
                print(
                    f"Device: {process_id}, Iteration: {iteration}/{len(subsets[process_id])}"
                )


def run():
    with open(os.path.join(os.path.dirname(__file__), 'config', "irn_inference.yaml")) as fp:
        config = yaml.full_load(fp)

    print("[ ", end="")
    multiprocessing.spawn(
        _work, nprocs=len(config["devices"]), args=(config,), join=True
    )
    print("]")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
