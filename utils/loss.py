import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_function(preds, targets):
    smooth = 1e-6
    #preds = nn.Sigmoid()(preds)
    n = preds.size(0)
    iflat = preds.view(n, -1)
    tflat = targets.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


class CharbonnierLoss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps) # torch.Size([3, 1, 512, 512])
        loss = torch.mean(error) # A number
        return loss