import numpy as np
import librosa
import scipy.signal as sgnl
import torch

from distance import SphericalAmbisonicsVisualizer
from distance import SphericalSourceVisualizer

# K-weighting filter to get perceived loudness as defined in https://www.itu.int/dms_pubrec/itu-r/rec/bs/r-rec-bs.1770-2-201103-s!!pdf-e.pdf
# uses the biquad coeffs given on pg 4 and 5
def k_weight_filter(audio, rate):
    # high-pass coeffs
    b_hp = np.array([1.0, -2.0, 1.0])
    a_hp = np.array([1.0, -1.99004745483398, 0.99007225036621])

    # high-shelf coeffs
    b_hs = np.array([1.5351248958697, -2.69169618940638, 1.19839281085285])
    a_hs = np.array([1.0, -1.69065929318241, 0.73248077421585])

    # apply filters
    filtered_audio = sgnl.lfilter(b_hp, a_hp, audio)
    filtered_audio = sgnl.lfilter(b_hs, a_hs, filtered_audio)

    return filtered_audio

# uses the same logic as arman's gen_sph_power_map to get frames, then stack them, then format them for an image CNN to read for embeddings
def ambi_to_tensor(ambiVis):
    frames = []
    for frame in ambiVis.loop_frames():
        if frame is None:
            break
        frames.append(frame)
    spatial = np.stack(frames, axis=0)
    spatial = spatial[:, None, :, :]
    spatial = spatial[None, ...]
    # gives (1,T,1,H,W) for image CNN
    return torch.from_numpy(spatial).float()

# cached mel banks to avoid recomputing them each call
_mel_banks = None

# getter for cached mel banks to use in extract features, will need to adjust logic if the parameters of the banks ever need to change
def get_mel_banks(sr, n_fft, n_mels, fmin, fmax):
    global _mel_banks
    if _mel_banks is None: 
        _mel_banks = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return _mel_banks

# takes wav and sample rate, uses armans ambiVis to get an image of the heatmap with a helper function and put it to a tensor, 
# then sums the channels mono, applies k-weight filter, gets short time fourier transform, computes rms, gets stft^2 for centroid, computes centroid, 
# gets cached mel banks (or if first run, creates and caches the mel banks), makes a mel spec, converts the mel spec to dB
# use dB mel to get onset, convert mel to normalized 0 - 1, then to tensor shape for CNN14 to get embeddings, returns dict of all features needed for analysis
# each scalar(rms, centroid, and onset) are given as float32 with tensor of (T,1)
def extract_features(wav, sr, position_fn, duration, angular_res=10., n_fft=512, hop_len=160, win_len=512, n_mels=64, fmin=20.0, fmax=20000.0):
    ambiVis = SphericalAmbisonicsVisualizer(wav, rate=sr, window=hop_len / sr, angular_res=angular_res)
    
    # Tensor shape is (1,T,1,H,W)
    spatial_tensor = ambi_to_tensor(ambiVis)

    sourceVis = SphericalSourceVisualizer(position_fn, duration, rate=1/hop_len, angular_res=angular_res)

    # getting pmaps per frame
    pos_maps = []
    for frame in sourceVis.loop_frames():
        pos_maps.append(frame)
    # Tensor shape is (1,T,1,H,W), stacking makes the shape (T,H,W)
    pos_maps = np.stack(pos_maps, axis=0)
    pos_maps = pos_maps[None, :, None, :, :]
    pos_maps = torch.from_numpy(pos_maps).float()

    y = librosa.to_mono(wav)
    y = k_weight_filter(y, sr)

    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_len, win_length=win_len))

    rms = librosa.feature.rms(S=stft).T
    rms = torch.from_numpy(rms).float()

    power_spec = stft ** 2
    centroid = librosa.feature.spectral_centroid(S=power_spec, sr=sr).T
    centroid = torch.from_numpy(centroid).float()

    mel_banks = get_mel_banks(sr, n_fft, n_mels, fmin, fmax)
    mel_spec = mel_banks.dot(power_spec)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)

    onset = librosa.onset.onset_strength(S=log_mel, sr=sr, hop_length=hop_len)
    onset = torch.from_numpy(onset[:, None]).float()

    log_mel = np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0).T
    # Tensor shape is (B,1,T,n_mels), matching the docs here: https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.Cnn14.html
    log_mel_tensor = torch.from_numpy(log_mel.copy()).unsqueeze(0).unsqueeze(0).float()
    return {"logmel": log_mel_tensor, "spatial": spatial_tensor, "positional": pos_maps, "rms": rms, "centroid": centroid, "onset": onset}