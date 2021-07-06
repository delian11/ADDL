import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_dim, d=64):
        super(Discriminator, self).__init__()
        lk = 0.01
        self.fc = nn.Sequential(
            nn.Linear(in_dim, d*4, bias=False),
            nn.LeakyReLU(lk, inplace=True),
            nn.Linear(d*4, d*2, bias=False),
            nn.LeakyReLU(lk, inplace=True),
            nn.Linear(d*2, d, bias=False),
            nn.LeakyReLU(lk, inplace=True),
            nn.Linear(d, 1, bias=False),
        )
    def forward(self, x):
        return self.fc(x)