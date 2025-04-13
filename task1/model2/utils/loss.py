import torch
import torch.nn.functional as F


def weighted_loss(pred, target, smooth=1e-5):
    pred = torch.clamp(pred, 1e-5, 1 - 1e-5)
    target = (target > 0.5).float()

    bce = F.binary_cross_entropy(pred, target)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = 1 - (2 * intersection + smooth) / (union + smooth)

    return 0.5 * bce + 0.5 * dice
