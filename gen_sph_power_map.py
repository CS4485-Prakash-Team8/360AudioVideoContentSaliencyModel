import sys, argparse
import numpy as np
import csv
import librosa
import scipy.signal as sgnl
import torch

from distance import SphericalAmbisonicsVisualizer, SphericalSourceVisualizer
from audio import load_wav
from video import VideoWriter
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from io import BytesIO
from PIL import Image

def run(input_fn, output_fn, position_fn='', angular_res='', csv_output='frame_data.csv'):
    
    data, rate = load_wav(input_fn)
    duration = data.shape[0] / float(rate)

    ambiVis = SphericalAmbisonicsVisualizer(data, rate, angular_res=angular_res)
    if position_fn:
        srcVis = SphericalSourceVisualizer(position_fn, duration, ambiVis.visualization_rate(), angular_res=angular_res)

    # Mel-spec vars are currently just set to what the Ambi visualizer is set to
    hop_length = int(ambiVis.window_frames)
    win_length = hop_length
    n_fft = win_length

    # Sum data to mono, k-weight filter it, short time fourier transform based on window given by ambivis, generate mel spec filter banks, place stft data in filter banks, change amplitude to dB
    mel_data = librosa.to_mono(data.T)
    mel_data = k_weight_filter(mel_data, rate)
    stft = np.abs(librosa.stft(mel_data, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False)) ** 2
    filter_banks = librosa.filters.mel(sr=rate, n_fft=n_fft, n_mels=20, fmin=20.0, fmax=20000.0)
    mel_spec = np.dot(filter_banks, stft)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)


    writer = VideoWriter(output_fn, video_fps=ambiVis.visualization_rate(), overwrite=True)
    mel_writer = VideoWriter('mel_output.mp4', video_fps=ambiVis.visualization_rate(), overwrite=True)

    cmap = np.stack(plt.get_cmap('inferno').colors)

    # Initialize vars for matplot used in mel-spec vid
    fig, ax = plt.subplots(figsize=(5,3), dpi=100)
    norm = Normalize(vmin=np.min(log_mel), vmax=np.max(log_mel))
    cax = ax.imshow(log_mel[:, :1], origin='lower', aspect='auto',  cmap='inferno', norm=norm)
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    
    with open(csv_output, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['Pixel_' + str(i) for i in range(1, 19 * 36 + 1)]
        csvwriter.writerow(header)
        
        frame_index = 0
        window_width = 30

        while True:
            frame = ambiVis.get_next_frame()
            if frame is None:
                break
            
            frame /= frame.max()

            if position_fn:
                frame += srcVis.get_next_frame()

            # Draw mel-spec frame
            if frame_index < log_mel.shape[1]:
                start_index = max(0, frame_index-window_width)
                mel_slice = log_mel[:, start_index:frame_index+1]
                cax.set_data(mel_slice)

                cax.set_extent((0, mel_slice.shape[1], 0, mel_slice.shape[0]))
                ax.set_xlim(0, window_width)
                ax.set_ylim(0, mel_slice.shape[0])
                fig.canvas.draw()

                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                mel_frame = np.array(Image.open(buf))
                buf.close()

                mel_writer.write_frame(mel_frame)

            # Normalize and apply colormap for video visualization
            frame_video = ((frame / frame.max()) * 255).astype(np.uint8)
            frame_video = (cmap[frame_video] * 255).astype(np.uint8)
            writer.write_frame(frame_video)

            # Flatten the frame to write into CSV
            frame_flattened = frame.flatten()
            csvwriter.writerow(frame_flattened)
            frame_index += 1

# K-weighting filter to get perceived loudness as defined in https://www.itu.int/dms_pubrec/itu-r/rec/bs/r-rec-bs.1770-2-201103-s!!pdf-e.pdf, using the biquad coeffs given on pg 4 and 5
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

# Sum data to mono, k-weight filter it, short time fourier transform based on window given in args,
# Generate mel spec filter banks, place stft data in filter banks, convert to dB, normalize for cnn(0-1), numpy to tensor for CNN, return tensor to float
# Just put placeholder defaults in for now 
def wav_to_logmel_tensor(wav, sr, bins=20, win_len=400, hop_len=400):
    duration = wav.shape[0] / float(sr)

    mel_data = librosa.to_mono(wav) #needed to transpose this for load_wav, but not sure if you will need it here, just make arg wav.T if so
    mel_data = k_weight_filter(mel_data, sr)
    
    #deprecated
    #stft = np.abs(librosa.stft(mel_data, n_fft=win_len, hop_length=hop_len, win_length=win_len)) ** 2
    #filter_banks = librosa.filters.mel(sr=sr, n_fft=win_len, n_mels=bins, fmin=20.0, fmax=20000.0)
    #mel_spec = np.dot(filter_banks, stft)
    
    mel_spec = librosa.feature.melspectrogram(y=mel_data, sr=sr, n_fft=win_len, hop_length=hop_len, n_mels=bins, fmin=20.0, fmax=20000.0, power=2.0)
    
    log_mel = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
    log_mel = np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0)
    log_mel_tensor = torch.from_numpy(log_mel.copy()).unsqueeze(0).unsqueeze(0)
    return log_mel_tensor.float()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_fn',    help='Input ambisonics filename.')
    parser.add_argument('output_fn',   help='Output video filename to store spherical power map.')
    parser.add_argument('--position_fn', default='', help='Ground-truth position file. Source locations will be superimposed.')
    parser.add_argument('--angular_res', default=10., type=float, help='Angular resolution.')
    return parser.parse_args(sys.argv[1:])

if __name__ == '__main__':
    args = parse_arguments()
    run(**vars(args))