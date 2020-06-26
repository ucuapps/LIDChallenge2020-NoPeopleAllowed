import torch.nn as nn
from utils.model_helpers import resnet
from utils.model_helpers._deeplab import DeepLabHeadV3Plus, DeepLabV3
from utils.model_helpers.utils import IntermediateLayerGetter


def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation,
    )

    inplanes = 2048
    low_level_planes = 256

    if name == "deeplabv3plus":
        return_layers = {"layer4": "out", "layer1": "low_level"}
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    if backbone.startswith("resnet"):
        model = _segm_resnet(
            arch_type,
            backbone,
            num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
    else:
        raise NotImplementedError
    return model


def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        "deeplabv3plus",
        "resnet50",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.model = deeplabv3plus_resnet50(
            num_classes=num_classes if num_classes != 2 else 1, pretrained_backbone=True
        )

    def forward(self, x):
        return {"out": self.model(x)}

    def get_params_groups(self):
        return (
            list(self.model.backbone.parameters()),
            list(self.model.classifier.parameters()),
        )
