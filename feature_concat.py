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

# temp yamnet for embeddings until the new one comes in, currently only outputs embedding around 2 times a second and repeats until next is generated
class YAMNetEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.model = yamnet.yamnet(pretrained=True).to(device).eval()
        self.converter = WaveformToInput()

    @torch.no_grad()
    def forward(self, wav, sr):
        x = self.converter(wav.float().to(self.device), sr)
        emb, _ = self.model(x)
        emb = emb.squeeze(0)
        if emb.ndim > 2:
            emb = emb.squeeze()
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        return emb

# concatenates yamn, spatial, and scalars from features to pass to LSTM with shape (T,concat_dim). We'll need to adjust dim sizes when we start seeing results
class AudioFeatureConcatenator(nn.Module):
    def __init__(self, yamn_dim=1024, spatial_dim=256, scalar_dim=3, concat_dim=256, device='cpu'):
        super().__init__()
        self.device = device
        self.yamn_encoder = YAMNetEncoder(device=device)
        self.spatial_encoder = SpatialEncoder(spatial_dim=spatial_dim).to(device)
        self.in_dim = yamn_dim + spatial_dim + scalar_dim
        self.proj = nn.Linear(self.in_dim, concat_dim)

    @torch.no_grad()
    def forward(self, features, sr=16000):
        wav = features["wav"]
        spatial = features["spatial"]
        rms = features["rms"]
        centroid = features["centroid"]
        onset = features["onset"]

        wav = wav.view(-1)
        n = wav.shape[0]
        if n < 15600:
            wav = F.pad(wav, (0, 15600 - n))
        wav = wav.unsqueeze(0)
        yam = self.yamn_encoder(wav, sr)
        T_yam = yam.shape[0]
        T_spatial = spatial.shape[0]

        rep = max(T_spatial // T_yam, 1)
        yam_rep = yam.repeat_interleave(rep, dim=0)

        if yam_rep.shape[0] < T_spatial:
            pad = T_spatial - yam_rep.shape[0]
            yam_rep = torch.cat([yam_rep, yam_rep[-1:].repeat(pad, 1)], dim=0)
        else:
            yam_rep = yam_rep[:T_spatial]

        spatial_emb = self.spatial_encoder(spatial)
        scalars = torch.cat([rms, centroid, onset], dim=-1)
        scalars = scalars.view(T_spatial, -1)

        concat = torch.cat([yam_rep, spatial_emb, scalars], dim=-1)
        return self.proj(concat)