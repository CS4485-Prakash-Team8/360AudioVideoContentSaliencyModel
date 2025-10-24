
# models/audio_cnn.py
# Minimal 2D CNN for log-Mel spectrograms (audio branch)
# Input: [B, C, T, F]  where C = channels (e.g., W,X,Y,Z), T=time frames, F=mel bins
# Output: embedding vector per clip + last conv feature map for visualization

import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, in_ch: int = 4, emb_dim: int = 256):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # T,F -> T/2, F/2
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # T,F -> T/4, F/4
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> [B,128,1,1]
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, x):
        # x: [B, C, T, F]
        z1 = self.block1(x)          # [B,32, T/2, F/2]
        z2 = self.block2(z1)         # [B,64, T/4, F/4]
        z3 = self.block3(z2)         # [B,128, T/4, F/4]
        pooled = self.gap(z3).flatten(1)  # [B,128]
        emb = self.proj(pooled)      # [B,emb_dim]
        return emb, z3               # Return embedding and last feature map
