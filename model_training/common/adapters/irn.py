import torch
import torch.nn as nn
from .base import ModelAdapter
from ..losses import AffinityDisplacementLoss
from ..datasets.irn.indexing import PathIndex

__all__ = ['IRNModelAdapter']


class IRNModelAdapter(ModelAdapter):
    def __init__(self, config, log_path):
        super(IRNModelAdapter, self).__init__(config, log_path)

        self.criterion = AffinityDisplacementLoss(
            PathIndex(radius=10, default_size=(config['train']['transform']['size'] // 4,
                                               config['train']['transform']['size'] // 4
                                               )))
        self.criterion = nn.DataParallel(self.criterion.to(self.device), device_ids=config['devices'])

    def forward(self, data):
        X = data['image']
        return self.model(X)

    def get_loss(self, y_pred, data):
        bg_pos_label = data['aff_bg_pos_label']
        fg_pos_label = data['aff_fg_pos_label']
        neg_label = data['aff_neg_label']

        loss = self.criterion(y_pred, y_true=(bg_pos_label, fg_pos_label, neg_label))
        return torch.mean(loss)

    def add_metrics(self, y_pred, data):
        pass
