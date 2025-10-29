# CNN for Audio 
import torch
import torch.nn as nn

class CNNAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),   # every batch gets normalized, more stable  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # halves H and W (kernel size)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),     
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),     
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),    
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        #allows any spectogram size to be input
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc  = nn.Linear(128, 1)              # size no longer depends on H×W
        self.out = nn.Sigmoid()  # sigmoid functinos turns score into probability

    #passing the filters one at a time
    def forward(self, x):
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.conv3(x)  
        x = self.conv4(x)  
        x = self.gap(x)      # average pooling
        x = x.flatten(1)     # flatten all dimensions except batch
        logits = self.linear2(x) #final output layer
        out = self.output(logits) #turns score -> probabilities between 0 and 1
        return out 

if __name__ == "__main__":
    
    ## NOT SURE WHERe TO GO FROM HERE FOR MAIN 

'''
just putting this here for reference on how to call the functions to get features

from distance import SphericalAmbisonicsVisualizer
from audio import load_wav

#arg to down sample to 16k
data, sr = load_wav(filename, 16000)

ambiVis = SphericalAmbisonicsVisualizer(data, rate = sr)

#gives tensors that contain the wav given(or chunk of wav given) at defined sample rate(if downsampled)
#Since I'm not 100% on what shape the tensors need to be for these, the formatting in the function may need to be changed
spatial = ambi_to_tensor(ambiVis)
#make sure that sr is what the pretrained expects and to set bins, win_len, and hop_len to what the pretrained expects as well
logmel = wav_to_logmel_tensor(data, sr)

'''