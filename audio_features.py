import numpy as np
import librosa
import torch
import resampy

from distance import AmbiVisMovingWindow

# uses the same logic as arman's gen_sph_power_map to get frames, then stack them, then format them for an image CNN to read for embeddings
def ambi_to_tensor(ambiVis):
    frames = []
    for frame in ambiVis.loop_frames():
        if frame is None:
            break
        frames.append(frame)
    spatial = np.stack(frames, axis=0)
    spatial = spatial[:, None, :, :]
    # gives (T,1,H,W) for image cnn(ResNet18)
    return torch.from_numpy(spatial).float()


# expects to take in wav and sample rate from load_wav at the moment, uses an altered ambiVis to get an image of the heatmap at stft rate with a helper function, 
# puts the heatmap to a tensor, then sums the channels mono, gets short time fourier transform, gets magnitudes, computes rms, gets stft^2 for centroid, computes centroid, 
# gets square of mags, computes onset with that, returns dict of all features needed for the 2 CNNs and concatenator
# each scalar(rms, centroid, and onset) are given as float32 with tensor of (T,1)
def extract_features(wav, rate, angular_res=10.0, n_mels=64, win_len=512, n_fft=512, hop_len=160, fmin=50.0, fmax=8000.0):
    sr = 16000.0

    if wav.ndim != 2 or wav.shape[1] != 4:
        raise ValueError(f"extract_features requires FOA (N,4). Got {wav.shape}")

    if rate != sr:
        y = resampy.resample(wav, rate, sr, axis=0, filter='kaiser_fast')
    else:
        y = wav

    ambiVis = AmbiVisMovingWindow(y, rate=sr, window=win_len, hop=hop_len, angular_res=angular_res)
    # Tensor shape is (T,1,H,W) 
    spatial_tensor = ambi_to_tensor(ambiVis)

    y = librosa.to_mono(y.T)

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_len, win_length=win_len, center=False)
    S, _ = librosa.magphase(stft)
    
    # not sure about using y over S here
    rms = librosa.feature.rms(y=y, frame_length=win_len, hop_length=hop_len, center=False).T
    rms = torch.from_numpy(rms).float()

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).T
    centroid = torch.from_numpy(centroid).float()

    power_spec = S**2
    onset = librosa.onset.onset_strength(S=power_spec, sr=sr, hop_length=hop_len, center=False)
    onset = torch.from_numpy(onset[:, None]).float()

    # Log-mel for ResNet-50 (wil need to go back to the old way of making logmel for CNN14 when this is tied in since most of the processes of this function are already done above)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_len, win_length=win_len, n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0, center=False)

    logmel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    logmel_norm = np.clip((logmel_db + 80.0) / 80.0, 0.0, 1.0).T

    # (N,) wav info for what yamnet expects
    y_ten = torch.from_numpy(y.astype("float32"))

    size_fix = min(spatial_tensor.shape[0], rms.shape[0], centroid.shape[0], onset.shape[0])
    spatial_tensor = spatial_tensor[:size_fix]
    rms = rms[:size_fix]
    centroid = centroid[:size_fix]
    onset = onset[:size_fix]
    logmel = torch.from_numpy(logmel_norm[:size_fix].copy()).unsqueeze(0).float()

    return {"wav": y_ten, "spatial": spatial_tensor, "rms": rms, "centroid": centroid, "onset": onset, "logmel": logmel}
