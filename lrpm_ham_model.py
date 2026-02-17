import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from ham import HAM
from torchvision.models import resnet18, ResNet18_Weights

class LRPM_HAM_Detector(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Lighter backbone (FAST for CPU)
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        modules = list(backbone.children())[:-2]
        self.backbone = torch.nn.Sequential(*modules)
        self.backbone.out_channels = 512

        # HAM attention
        self.ham = HAM(512)

        # Anchors
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        self.model = FasterRCNN(
            self.backbone,
            num_classes=2,   # background + polyp
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
     # ✅ ADD THIS METHOD
    def extract_feature_maps(self, images):
        x = torch.stack(images)      # [B,3,H,W]
        features = self.backbone(x)  # CNN feature mapping
        features = self.ham(features)
        return features
    
    def forward(self, images, targets=None):
        return self.model(images, targets)
