"""Lightweight PPG encoder used in the primary regional-weighting experiment."""

import torch.nn.functional as F
from torch import nn


class PPGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 9, 2, 4), nn.ReLU(),
            nn.Conv1d(32, 64, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(64, 128, 5, 2, 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, 128)
        self.head = nn.Linear(128, 2)

    def forward(self, x):
        representation = F.normalize(self.proj(self.net(x).squeeze(-1)), dim=1)
        return representation, self.head(representation)
