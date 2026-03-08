import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
import random
import matplotlib.pyplot as plt


class LBPLoss(torch.nn.Module):

    def __init__(self, tau=5, margin=0):
        super().__init__()

        self.margin = margin
        self.tau = tau

        kernels = torch.tensor(
            [
                [[[1, 0, 0], [0, -1, 0], [0, 0, 0]]],
                [[[0, 1, 0], [0, -1, 0], [0, 0, 0]]],
                [[[0, 0, 1], [0, -1, 0], [0, 0, 0]]],
                [[[0, 0, 0], [1, -1, 0], [0, 0, 0]]],
                [[[0, 0, 0], [0, -1, 1], [0, 0, 0]]],
                [[[0, 0, 0], [0, -1, 0], [1, 0, 0]]],
                [[[0, 0, 0], [0, -1, 0], [0, 1, 0]]],
                [[[0, 0, 0], [0, -1, 0], [0, 0, 1]]],
            ],
            dtype=torch.float32,
        )

        self.register_buffer("kernels", kernels)

    def to_gray(self, x):
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def forward(self, fake, real):

        fake = self.to_gray(fake)
        real = self.to_gray(real)

        fake_rsp = F.conv2d(fake, self.kernels, padding=1)
        real_rsp = F.conv2d(real, self.kernels, padding=1)

        diff = self.tau * (fake_rsp - real_rsp).abs()

        penalty = F.relu(diff - self.margin)

        return (penalty**2).mean()
