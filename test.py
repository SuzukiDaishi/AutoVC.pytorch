from synthesis import build_model
from synthesis import wavegen
import torch
import numpy as np
import math
from utils.dsp import load_wav
from utils.dsp import melspectrogram
import time


if __name__ == '__main__':

    dim_neck = 32
    dim_emb = 256
    dim_pre = 512
    freq = 32

    device = torch.device('cpu')
    wavnet = build_model().to(device)
    checkpoint = torch.load('./checkpoint_step001000000_ema.pth', map_location=device)
    wavnet.load_state_dict(checkpoint["state_dict"])

    wav = load_wav('./VCTK-Corpus/wav48/p225/p225_001.wav')[:16000]

    start_time = time.time()

    # print(wav.shape)
    mel = melspectrogram(wav)
    pad_len = math.ceil(mel.shape[1] / 32) * 32 - mel.shape[1]
    mel = np.pad(mel, ((0,0), (0, pad_len)), mode='constant')

    mel_rec = mel

    mel_rec = mel_rec[:,:-pad_len]

    # print(mel_rec.shape)

    c = np.transpose(mel_rec, (1, 0))
    waveform = wavegen(wavnet, device, c=c)
    # print(waveform)

    print( time.time() - start_time )
    
