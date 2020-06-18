import numpy as np
import torch
import matplotlib.pyplot as plt

from .base import ModelAdapter

__all__ = ['ClassificationModelAdapter']


class ClassificationModelAdapter(ModelAdapter):
    def make_tensorboard_grid(self, batch_sample):
        data, y_pred = batch_sample['data'], batch_sample['y_pred']
        X, y = data[0], data[1]
        X = self.denormalize(X).cpu().numpy()
        fig, axes = plt.subplots(1, X.shape[0], figsize=(30, 10))
        for i, ax in enumerate(axes):
            ax.imshow(X[i].transpose(1, 2, 0))
            ax.axis('off')
            y_i, y_pred_i = y[i], y_pred[i]  # TODO: y_pred['out'][i]
            y_i = list(torch.where(y_i == 1)[0].cpu().numpy())
            y_pred_i = list(torch.where(y_pred_i > 0.5)[0].cpu().numpy())
            ax.title.set_text(f'T:{y_i}, P: {y_pred_i}')
        fig.canvas.draw()
        return np.array(fig.canvas.renderer.buffer_rgba())[:, :, :-1].transpose(2, 0, 1)  # remove a channel

    def get_loss(self, y_pred, data):
        loss = super(ClassificationModelAdapter, self).get_loss(y_pred, data)
        return loss.sum(dim=1).mean()
