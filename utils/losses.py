import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


def mask_loss(masks1, masks2):
    n = len(masks1)
    sum = 0
    for i in range(n):
        mask1 = masks1[i]
        mask2 = masks2[i]
        # mask1_logit = F.softmax(mask1, dim=-1)
        # mask2_logit = F.softmax(mask2, dim=-1)
        mask1_logit = mask1 / torch.norm(mask1)
        mask2_logit = mask2 / torch.norm(mask2)

        sum += torch.norm(mask1_logit - mask2_logit)
    return sum


def kl_loss(feat_real, feat_fit):
    # kl = nn.KLDivLoss(size_average=None)
    real_logit = F.softmax(feat_real, dim=-1)
    _kl_1 = torch.sum(real_logit * (F.log_softmax(feat_real, dim=-1) - F.log_softmax(feat_fit, dim=-1)), 1)
    kl_pos = torch.mean(_kl_1)

    return kl_pos
