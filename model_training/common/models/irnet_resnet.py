import torch
import torch.nn as nn
import torchvision
from copy import deepcopy


class IRNet(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(IRNet, self).__init__()

        self.frozen_backbone = freeze_backbone
        backbone = torchvision.models.resnet.resnet50(pretrained=True,
                                                      replace_stride_with_dilation=[False, False, True])
        self.stage1 = nn.Sequential(
            deepcopy(backbone.conv1),
            deepcopy(backbone.bn1),
            deepcopy(backbone.relu),
            deepcopy(backbone.maxpool)
        )
        self.stage2 = deepcopy(backbone.layer1)
        self.stage3 = deepcopy(backbone.layer2)
        self.stage4 = deepcopy(backbone.layer3)
        self.stage5 = deepcopy(backbone.layer4)

        # branch: class boundary detection
        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

        # branch: displacement field
        self.fc_dp1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.fc_dp2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
        )
        self.fc_dp3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )
        self.fc_dp4 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_dp5 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_dp6 = nn.Sequential(
            nn.Conv2d(768, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_dp7 = nn.Sequential(
            nn.Conv2d(448, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1, bias=False),
            nn.InstanceNorm2d(num_features=2, affine=False, track_running_stats=True)  # to normalize displacement field
        )

        self.backbone = [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5]
        self.edge_layers = [self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6]
        self.dp_layers = [self.fc_dp1, self.fc_dp2, self.fc_dp3, self.fc_dp4, self.fc_dp5, self.fc_dp6, self.fc_dp7]

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for layer in self.backbone:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        # backbone
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        # boundary detection
        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)
        edge4 = self.fc_edge4(x4)
        edge5 = self.fc_edge5(x5)
        edge_out = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))

        # displacement field
        dp1 = self.fc_dp1(x1)
        dp2 = self.fc_dp2(x2)
        dp3 = self.fc_dp3(x3)
        dp4 = self.fc_dp4(x4)
        dp5 = self.fc_dp5(x5)

        dp_up3 = self.fc_dp6(torch.cat([dp3, dp4, dp5], dim=1))
        dp_out = self.fc_dp7(torch.cat([dp1, dp2, dp_up3], dim=1))

        return edge_out, dp_out

    def get_params_groups(self):
        if self.frozen_backbone:
            return [p for module in self.edge_layers for p in module.parameters()], \
                   [p for module in self.dp_layers for p in module.parameters()],
        else:
            return [p for module in self.backbone for p in module.parameters()], \
                   [p for module in self.edge_layers for p in module.parameters()], \
                   [p for module in self.dp_layers for p in module.parameters()],


if __name__ == '__main__':
    model = IRNet(freeze_backbone=True)
    print(list(model.parameters()))

    X = torch.randn(1, 3, 224, 224)
    y, z = model(X)

    print(y.shape, z.shape)
