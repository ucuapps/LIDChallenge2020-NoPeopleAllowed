import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    """
    Implementation of mean soft-IoU loss for semantic segmentation
    """
    __EPSILON = 1e-6

    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
        y_pred: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        y_true: a tensor of shape [B, H, W].
        Returns:
        float: soft-iou loss.
        """
        num_classes = y_pred.shape[1]

        y_true_dummy = torch.eye(num_classes)[y_true.squeeze(1)]
        y_true_dummy = y_true_dummy.permute(0, 3, 1, 2).to(y_true.device)

        y_pred_proba = F.softmax(y_pred, dim=1)

        intersection = torch.sum(y_pred_proba * y_true_dummy, dim=(2, 3))
        union = torch.sum(y_pred_proba ** 2 + y_true_dummy ** 2, dim=(2, 3)) - intersection
        iou_loss = ((intersection + self.__EPSILON) / (union + self.__EPSILON))

        return 1 - iou_loss.mean()


class DiceLossWithLogits(nn.Module):
    @staticmethod
    def dice_metric(input, target):
        smooth = 1.0
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

    def forward(self, input, target):
        input = (torch.sigmoid(input) > 0.5).type(torch.float32)
        return 1 - self.dice_metric(input, target)


class AffinityDisplacementLoss(nn.Module):
    path_indices_prefix = "path_indices"

    def __init__(self, path_index):

        super(AffinityDisplacementLoss, self).__init__()

        self.path_index = path_index

        self.n_path_lengths = len(path_index.path_indices)
        for i, pi in enumerate(path_index.path_indices):
            self.register_buffer(AffinityDisplacementLoss.path_indices_prefix + str(i), torch.from_numpy(pi))

        self.register_buffer(
            'disp_target',
            torch.unsqueeze(torch.unsqueeze(torch.from_numpy(path_index.search_dst).transpose(1, 0), 0), -1).float())

    def to_affinity(self, edge):
        aff_list = []
        edge = edge.view(edge.size(0), -1)

        for i in range(self.n_path_lengths):
            ind = self._buffers[AffinityDisplacementLoss.path_indices_prefix + str(i)]
            ind_flat = ind.view(-1)
            dist = torch.index_select(edge, dim=-1, index=ind_flat)
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)
        aff_cat = torch.cat(aff_list, dim=1)

        return aff_cat

    def to_pair_displacement(self, disp):
        height, width = disp.size(2), disp.size(3)
        radius_floor = self.path_index.radius_floor

        cropped_height = height - radius_floor
        cropped_width = width - 2 * radius_floor

        disp_src = disp[:, :, :cropped_height, radius_floor:radius_floor + cropped_width]

        disp_dst = [disp[:, :, dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
                    for dy, dx in self.path_index.search_dst]
        disp_dst = torch.stack(disp_dst, 2)

        pair_disp = torch.unsqueeze(disp_src, 2) - disp_dst
        pair_disp = pair_disp.view(pair_disp.size(0), pair_disp.size(1), pair_disp.size(2), -1)

        return pair_disp

    def to_displacement_loss(self, pair_disp):
        return torch.abs(pair_disp - self.disp_target)

    def forward(self, y_pred, y_true):
        edge_out, dp_out = y_pred
        bg_pos_label, fg_pos_label, neg_label = y_true

        aff = self.to_affinity(torch.sigmoid(edge_out))
        pos_aff_loss = (-1) * torch.log(aff + 1e-5)
        neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)

        pair_disp = self.to_pair_displacement(dp_out)
        dp_fg_loss = self.to_displacement_loss(pair_disp)
        dp_bg_loss = torch.abs(pair_disp)

        bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
        fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)
        pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
        neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

        dp_fg_loss = torch.sum(dp_fg_loss * torch.unsqueeze(fg_pos_label, 1)) / (2 * torch.sum(fg_pos_label) + 1e-5)
        dp_bg_loss = torch.sum(dp_bg_loss * torch.unsqueeze(bg_pos_label, 1)) / (2 * torch.sum(bg_pos_label) + 1e-5)

        total_loss = (pos_aff_loss + neg_aff_loss) / 2 + (dp_fg_loss + dp_bg_loss) / 2

        return total_loss


def get_loss(loss_config):
    loss_name = loss_config['name']
    if loss_name == 'categorical_cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=loss_config.get('ignore_index', -1))
    elif loss_name == 'binary_cross_entropy':
        return nn.BCEWithLogitsLoss(reduction='none')

    elif loss_name == 'mean_iou':
        return IoULoss()
    elif loss_name == 'kldiv':
        return nn.KLDivLoss(reduction='none')
    elif loss_name == 'binary_dice':
        return DiceLossWithLogits()
    else:
        raise ValueError(f"Loss [{loss_name}] not recognized.")
