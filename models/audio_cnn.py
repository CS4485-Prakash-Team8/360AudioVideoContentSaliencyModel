
# models/audio_cnn.py
# Minimal 2D CNN for log-Mel spectrograms (audio branch)
# Input: [B, C, T, F]  where C = channels (e.g., W,X,Y,Z), T=time frames, F=mel bins
# Output: embedding vector per clip + last conv feature map for visualization

import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, in_ch: int = 4, emb_dim: int = 256):
        super().__init__()
         ## convolution layers

        ## kernel sizes? 
        ##  pooling 


        ## flattening


        ##output
