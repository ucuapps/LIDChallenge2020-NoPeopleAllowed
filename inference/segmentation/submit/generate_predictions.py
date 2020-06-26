import yaml
import numpy as np
import os
import cv2
import click
import torch
import torch.nn.functional as F
import tqdm
import ttach as tta

from utils import ImageNetSegmentationTest, get_network, get_transforms


def tta_model_predict(X, model):
    tta_transforms = tta.Compose([tta.HorizontalFlip(), tta.Scale(scales=[0.5, 1, 2])])
    masks = []
    for transformer in tta_transforms:
        augmented_image = transformer.augment_image(X)
        model_output = model(augmented_image)["out"]

        deaug_mask = transformer.deaugment_mask(model_output)
        masks.append(deaug_mask)

    mask = torch.sum(torch.stack(masks), dim=0) / len(masks)
    return mask


@click.command()
@click.argument("save_output_path", type=click.Path())
@click.argument("data_path", type=click.Path())
@click.option("--device", type=str, default="cuda:0")
def run_inference(save_output_path, data_path, device):
    experiment_path = "best_model/"

    with open(os.path.join(experiment_path, "config.yaml")) as config_file:
        config = yaml.full_load(config_file)

    config["experiment_path"] = experiment_path
    config["model"]["weights_path"] = "last_model.h5"

    config["save_outs"] = True
    config["use_tta"] = True
    config["save_output_path"] = save_output_path

    if not os.path.exists(save_output_path):
        os.makedirs(save_output_path)

    test_transform = get_transforms(config["val"]["transform"])
    test_ds = ImageNetSegmentationTest(data_path, transform=test_transform)

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=True, num_workers=12, drop_last=True
    )

    model_path = os.path.join(
        config["experiment_path"], config["model"]["weights_path"]
    )
    model = get_network(config["model"])
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for X, name, orig_im in tqdm.tqdm(test_dl):
            list_saved_names = os.listdir(config["save_output_path"])

            if name[0] + ".png" not in list_saved_names:
                X = X.to(device)

                if config["use_tta"]:
                    y_pred = tta_model_predict(X, model)
                else:
                    y_pred = model(X)["out"]

                y_pred = F.interpolate(y_pred, orig_im.size()[1:-1], mode="nearest")

                if config["save_outs"]:
                    img_pred = np.argmax(y_pred.cpu().numpy().squeeze(), axis=0)
                    cv2.imwrite(
                        os.path.join(config["save_output_path"], name[0] + ".png"),
                        img_pred.astype(np.uint8),
                    )

                del X
                del y_pred
                del orig_im
                torch.cuda.empty_cache()

    print("Inference completed!")


if __name__ == "__main__":
    run_inference()
