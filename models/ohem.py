import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class SmoothOhemLoss(nn.Module):
    def __init__(self):
        super(SmoothOhemLoss, self).__init__()
        self.smooth_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, preds_imgs, gt_imgs, ignore_masks, gt_weights):
        loss_every_sample = []
        batc_size = preds_imgs.size(0)
        for i in range(batc_size):
            pred_img = preds_imgs[i].view(1, -1)
            gt_img = gt_imgs[i].view(1, -1)
            ignore_mask = ignore_masks[i].view(1, -1)
            gt_weight = gt_weights[i].view(1, -1)
            positive_mask = (gt_img > 0).float()
            negetive_mask = (gt_img == 0).float() * ignore_mask.float()
            sample_loss = self.smooth_loss(pred_img, gt_img) * gt_weight

            positive_loss = torch.masked_select(sample_loss, positive_mask.byte())
            negative_loss = torch.masked_select(sample_loss, negetive_mask.byte())
            num_positive = int(positive_mask.sum().data.cpu().item())

            k = num_positive * 3
            num_all = torch.sum(ignore_mask)
            if num_positive > 0:
                if k+num_positive > num_all:
                    k = int(num_all-num_positive)
                negative_loss_topk, _ = torch.topk(negative_loss, k)
                avg_sample_loss = positive_loss.mean() + negative_loss_topk.mean()
            else:
                negative_loss_topk, _ = torch.topk(negative_loss, 500)
                avg_sample_loss = negative_loss_topk.mean()

            loss_every_sample.append(avg_sample_loss)
        try:
            torch.stack(loss_every_sample, 0)
        except:
            print('a')
        return torch.stack(loss_every_sample, 0).mean()
