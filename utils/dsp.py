import numpy as np
import librosa, math

from numpy.lib.type_check import imag, real
from hparams import hparams as hp

def load_wav(filename):
    x = librosa.load(filename, sr=hp.sample_rate)[0]
    return x

def save_wav(y, filename) :
    librosa.output.write_wav(filename, y, hp.sample_rate)
    
mel_basis = None

def linear_to_mel(spectrogram):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)

def build_mel_basis():
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax)

def normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)

def unnormalize(S):
    return S * -hp.min_level_db + hp.min_level_db

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
    return np.power(10.0, x * 0.05)

def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)

def stft(y):
    return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)

def complex_linear_to_mel(spectrogram):
    spct_r = spectrogram[0]
    spct_i = spectrogram[1]
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.array([np.dot(mel_basis, spct_r), np.dot(mel_basis, spct_i)])

def complex_amp_to_db(x):
    yr = 20 * np.log10(np.maximum(1e-5, x[0]))
    yi = 20 * np.log10(np.maximum(1e-5, x[1]))
    return np.array([yr, yi])

def complex_normalize(S):
    yr = np.clip((S[0] - hp.min_level_db) / -hp.min_level_db, 0, 1)
    yi = np.clip((S[1] - hp.min_level_db) / -hp.min_level_db, 0, 1)
    return np.array([yr, yi])

def split_complex(x):
    return np.array([x.real, x.imag])

def stft_normalize(S):
    yr = np.clip((S[0] - hp.stft_real_min) / (hp.stft_real_max-hp.stft_real_min), 0, 1)
    yi = np.clip((S[1] - hp.stft_real_min) / (hp.stft_real_max-hp.stft_real_min), 0, 1)
    return np.array([yr, yi])

def complex_melspectrogram(y):
    D = stft(y)
    S = complex_amp_to_db( complex_linear_to_mel( split_complex(D) ) ) # dbやめたほうが...
    return complex_normalize(S)

def complex_stft(y):
    D = stft(y)
    S = split_complex(D)
    return stft_normalize(S)