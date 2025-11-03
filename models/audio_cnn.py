## ResNet-50 pretrained model, takes in sensor from audio_features and returns the proper embeddings to feed into LSTM and concat
## NEXT STEP BEING FEEDING INTO SPATIAL ENCODER ( CONCAT ) 
## to run : Python -m models.audio_cnn samplespatialaudio/1.wav
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from audio import load_wav
from audio_features import extract_features
class AudioResNet50(nn.Module):
    """
    Input:  logmel tensor (B,1,T,M)
        b = batch size
        t = time frames
        m = mel bins
    Output: (B, T, D)  — one embedding per time frame 
        FRAMWISE EMBBEDDING
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()  # initializing mdoule for pretrained ResNet-50 model
        # loading the pretrained weights from ResNet-50 
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # change first conv. layerto accept 1-channel log mel
        old = base.conv1.weight  # (64,3,7,7)
        conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 2), padding=(3, 3), bias=False)

         # avg the 3 RGB weights into 1 mono channel to reuse pretrained filters
        with torch.no_grad():
            conv1.weight.copy_(old.mean(dim=1, keepdim=True))
        base.conv1 = conv1 # replacing with new layer

        # MaxPool: downsample only frequency
        base.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        # Ensure deeper stages downsample only in frequency (never time)
        self._freq_only_stride(base.layer2)
        self._freq_only_stride(base.layer3)
        self._freq_only_stride(base.layer4)
        self.trunk = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )
        self.proj = nn.Linear(2048, out_dim)

    @staticmethod
    def _freq_only_stride(layer: nn.Sequential):
        # Accessing the 1st bottleneck block in the ResNet layer
        b0 = layer[0]
        # If conv2 has stride (2,2), change it to (1,2) — no downsampling in time
        if hasattr(b0, "conv2") and tuple(b0.conv2.stride) == (2, 2):
            b0.conv2.stride = (1, 2)
        if b0.downsample is not None and tuple(b0.downsample[0].stride) == (2, 2):
            b0.downsample[0].stride = (1, 2)

    ## now we modified ResNett on the tensor
    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        # logmel: (B,1,T,M)
        x = self.trunk(logmel)      # Run CNN backbone -> (B, 2048, T, M')
        x = x.mean(dim=3)           # avg over frequency -> (B, 2048, T)
        x = x.transpose(1, 2)       # Move time dimension to front -> (B, T, 2048)
        return self.proj(x)         #  Linear projection -> (B, T, 256)
def main():
    if len(sys.argv) < 2:
        print("Usage: python -m models.audio_cnn <path_to_wav>")
        sys.exit(1)
    wav_path = sys.argv[1]
    wav, sr = load_wav(wav_path, rate=16000)             # (samples, channels)
    duration = wav.shape[0] / float(sr)
    # Get tensors (logmel, spatial, scalars, .)
    # no position file expected
    print(" Running extract_features (ambisonics + audio features)...")
   
    feats = extract_features(
        wav=wav, sr=sr, position_fn="", duration=duration,
        angular_res=10.0, n_fft=512, hop_len=160, win_len=512,
        n_mels=64, fmin=50.0, fmax=8000.0
    )
   
    #CONFIRMING values are being called right
    logmel = feats["logmel"]                        # (1, T, M)
    spatial = feats["spatial"]                      # (T, 1, H, W)
    rms = feats["rms"]                              # (T, 1)
    centroid = feats["centroid"]                    # (T, 1)
    onset = feats["onset"]                          # (T, 1)
    print(f"[SENSOR] logmel {tuple(logmel.shape)}, "
          f"spatial {tuple(spatial.shape)}, "
          f"rms {tuple(rms.shape)}, centroid {tuple(centroid.shape)}, onset {tuple(onset.shape)}")
    # model input has to be (1,1,T,M)
    logmel = feats["logmel"]
    if logmel.ndim == 3:   # (1,T,M) -> (1,1,T,M)
        logmel = logmel.unsqueeze(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioResNet50(out_dim=256).to(device).eval()
    with torch.no_grad():
        emb = model(logmel.float().to(device))   # (1, T, 256)
    print(f"[OUTPUT] Embeddings shape: {tuple(emb.shape)}")
    torch.save(emb.cpu(), "audio_embeddings.pt")
   
if __name__ == "__main__":
    main()

