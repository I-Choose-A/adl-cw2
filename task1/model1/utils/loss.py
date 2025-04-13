import torch
import torch.nn.functional as F


def weighted_loss(pred, target, smooth=1e-5):
    pred = torch.clamp(pred, 1e-5, 1 - 1e-5)
    target = (target > 0.5).float()

    bce = F.binary_cross_entropy(pred, target)

    dims = tuple(range(1, pred.dim()))  # exclude batch dim
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = 1 - (2 * intersection + smooth) / (union + smooth)
    dice = dice.mean()

    bce_no_reduction = F.binary_cross_entropy(pred, target, reduction='none')  # per sample BCE
    p_t = target * pred + (1 - target) * (1 - pred)  # probability of the true class
    # Focal Loss calculation
    focal = (0.25 * (1 - p_t) ** 2.0 * bce_no_reduction).mean()  # apply focal scaling

    return 0.33 * bce + 0.33 * dice + 0.34 * focal
