import torch
import torch.nn as nn

class HAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x)
        x = x * ca

        # Spatial attention
        sa = self.spatial_att(x)
        x = x * sa

        return x
