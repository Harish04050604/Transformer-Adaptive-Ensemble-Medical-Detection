#loss.py

import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.CrossEntropyLoss()

    def forward(self, logits, masks):
        bce_loss     = self.bce(logits, masks.squeeze(1).long())

        probs        = F.softmax(logits, dim=1)[:, 1]
        masks_f      = masks.squeeze(1).float()
        intersection = (probs * masks_f).sum(dim=(1, 2))
        dice_loss    = 1 - (2 * intersection + 1) / \
                       (probs.sum(dim=(1, 2)) + masks_f.sum(dim=(1, 2)) + 1)

        return bce_loss + dice_loss.mean()