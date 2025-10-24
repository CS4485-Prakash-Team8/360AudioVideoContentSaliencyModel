# scripts/run_audio_cnn_demo.py
# Run from the PROJECT ROOT:
#   python -m scripts.run_audio_cnn_demo --wav SampleSpatialAudio\1.wav --sr 16000 --n_mels 64

import os, sys, importlib.util

# --- Paths ---
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))     # ...\project\scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                    # ...\project
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- FORCE-LOAD the stdlib 'cmd' module so pdb/torch use the real one, not your local cmd.py ---
# Works in and out of virtualenvs: stdlib lives under sys.base_prefix\Lib
STDLIB_CMD_PATH = os.path.join(sys.base_prefix, "Lib", "cmd.py")
if os.path.isfile(STDLIB_CMD_PATH):
    spec = importlib.util.spec_from_file_location("cmd", STDLIB_CMD_PATH)
    std_cmd = importlib.util.module_from_spec(spec)
    sys.modules["cmd"] = std_cmd  # register stdlib cmd as the 'cmd' module
    spec.loader.exec_module(std_cmd)
# If that file isn’t found, we’ll fall back to normal import resolution.

# --- now safe to import torch (this is where pdb -> cmd happens) ---
import torch

# --- normal imports (your code) ---
import argparse, numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import librosa
from models.audio_cnn import AudioCNN  # requires models/audio_cnn.py and models/__init__.py

def wav_to_logmel(x, sr, n_mels=64, win=400, hop=160):
    S = librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=win, hop_length=hop, win_length=win,
        n_mels=n_mels, power=2.0
    )
    return np.log(np.maximum(S, 1e-8)).astype(np.float32)  # [mel, frames]

def load_logmel_stacked(wav_path, sr, n_mels):
    y, sr_out = librosa.load(wav_path, sr=sr, mono=False)  # mono=False keeps channels
    if y.ndim == 1:            # mono
        y = y[None, :]         # -> [C=1, T]
    C, _T = y.shape
    mels = [wav_to_logmel(y[c], sr=sr_out, n_mels=n_mels) for c in range(C)]  # each [F,T]
    mels = np.stack(mels, axis=0)        # [C,F,T]
    mels = np.transpose(mels, (0, 2, 1)) # [C,T,F]
    x = torch.from_numpy(mels).unsqueeze(0)  # [1,C,T,F]
    return x, C, sr_out

def save_feature_grid(feat_zt, out_png, nrow=8, tile_w=16):
    # feat_zt: [C, T', F']  (channels, time, mel)
    import torch
    t_mid = feat_zt.shape[1] // 2
    maps = feat_zt[:, t_mid, :]  # [C, F']
    maps = (maps - maps.min(dim=1, keepdim=True).values) / (
        maps.max(dim=1, keepdim=True).values - maps.min(dim=1, keepdim=True).values + 1e-8
    )
    maps = maps.unsqueeze(1).unsqueeze(-1)   # [C,1,F',1]
    maps = maps.repeat(1, 1, 1, tile_w)     # widen for visibility
    grid = vutils.make_grid(maps[:min(32, maps.shape[0])], nrow=nrow, pad_value=1.0)
    plt.imsave(out_png, grid.permute(1,2,0).numpy())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Path to WAV (mono or multichannel/ambisonic)")
    ap.add_argument("--sr", type=int, default=16000, help="Resample rate for mel")
    ap.add_argument("--n_mels", type=int, default=64, help="Mel bins")
    ap.add_argument("--emb_dim", type=int, default=256, help="Embedding size")
    ap.add_argument("--outdir", default="outputs", help="Where to save visualizations")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x, C, sr = load_logmel_stacked(args.wav, args.sr, args.n_mels)  # [1,C,T,F]
    print(f"Loaded log-mel: x.shape={tuple(x.shape)}, channels={C}, sr={sr}")
    x = x.to(device)

    model = AudioCNN(in_ch=C, emb_dim=args.emb_dim).to(device).eval()

    with torch.no_grad():
        emb, feat = model(x)  # emb: [1,emb_dim], feat: [1,128,T',F']
    print(f"Embedding shape: {tuple(emb.shape)}")
    print(f"Feature map shape: {tuple(feat.shape)}")

    np.save(os.path.join(args.outdir, "embedding.npy"), emb.cpu().numpy())
    feat_cf = feat.squeeze(0).cpu()  # [128, T', F']
    save_feature_grid(feat_cf, os.path.join(args.outdir, "audio_feat_grid.png"))
    print("Saved outputs to:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
