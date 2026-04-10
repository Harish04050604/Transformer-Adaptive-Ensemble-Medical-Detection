#transformer_model.py

import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class PolypSegformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=2,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits  = outputs.logits
        logits  = F.interpolate(logits, size=x.shape[-2:],
                                mode="bilinear", align_corners=False)
        return logits