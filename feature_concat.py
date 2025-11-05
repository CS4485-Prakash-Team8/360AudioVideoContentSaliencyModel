import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from torch_vggish_yamnet import yamnet
from torch_vggish_yamnet.input_proc import WaveformToInput

# resnet18 encoder for spatial, uses imagenet weights and sets the first layer to correct channel and use weights, 
# takes (T,1,H,W), then conv gives (T,512,1,1), and compresses to (T,256)
class SpatialEncoder(nn.Module):
    def __init__(self, spatial_dim=256):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        old_w = base.conv1.weight
        conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv.weight = nn.Parameter(old_w.mean(dim=1, keepdim=True))
        base.conv1 = conv
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Linear(512, spatial_dim)

    def forward(self, x):
        feats = self.encoder(x)
        feats = feats.squeeze(-1).squeeze(-1)
        return self.proj(feats)

# concatenates yamn, spatial, and scalars from features to pass to LSTM with shape (T,concat_dim). We'll need to adjust dim sizes when we start seeing results
##  implemented resnet-50 below
class AudioFeatureConcatenator(nn.Module):
    def __init__(
        self,
        audio_dim: int = 256,    # output dim of AudioResNet50
        spatial_dim: int = 256,  # output dim of SpatialEncoder
        scalar_dim: int = 3,     # rms + centroid + onset
        concat_dim: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.audio_encoder = AudioResNet50(out_dim=audio_dim).to(device).eval()
        self.spatial_encoder = SpatialEncoder(spatial_dim=spatial_dim).to(device)
        self.in_dim = audio_dim + spatial_dim + scalar_dim
        self.proj = nn.Linear(self.in_dim, concat_dim)

    @torch.no_grad()
    def forward(self, features: dict, sr: int = 16000):
        # unpack
        logmel   = features["logmel"]          # (1,T,M)
        spatial  = features["spatial"]         # (T,1,H,W)
        rms      = features["rms"]             # (T,1)
        centroid = features["centroid"]        # (T,1)
        onset    = features["onset"]           # (T,1)

        # audio path (ResNet-50 expects (B,1,T,M))
        if logmel.ndim == 3:
            logmel = logmel.unsqueeze(1)       # (1,1,T,M)


        audio_emb = self.audio_encoder(logmel.to(self.device, dtype=torch.float32))  # (1,T,256)
        audio_emb = audio_emb.squeeze(0)       # removing batch dimension (1,T,256) -> (T,256)

        # spatial path
        spatial_emb = self.spatial_encoder(spatial.to(self.device, dtype=torch.float32))  # (T,256)

        # scalar features
        scalars = torch.cat([rms, centroid, onset], dim=-1).to(self.device, dtype=torch.float32)  # (T,3)


        # fuse
        concat = torch.cat([audio_emb, spatial_emb, scalars], dim=-1)  # (T, 256+256+3)
        fused  = self.proj(concat)                                     # (T, concat_dim)

        return fused
