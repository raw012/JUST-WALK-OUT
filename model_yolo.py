# This file aims to rebuild the model yolo
import torch
import torch.nn as nn

class YOLOModel(nn.Module):
    def __init__(self, S=7, B=2, C=3):
        super().__init__()
        self.S, self.B, self.C = S, B, C

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((S, S))
        )

        self.head = nn.Conv2d(
            64, B * 5 + C, kernel_size=1
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x.permute(0, 2, 3, 1)  # [N,S,S,B*5+C]
