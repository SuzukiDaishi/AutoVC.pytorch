import pyworld
import numpy as np
from hparams import hparams as hp

def world_decompose(wav, fs, frame_period=5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, timeaxis, sp, ap

def world_encode_spectral_envelop(sp, fs, dim=24):
    # Get Mel-Cepstral coefficients (MCEPs)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    return coded_sp

def world_encode_data(wave, fs, frame_period=5.0, coded_dim=24):
    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()
    for wav in wave:
        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=fs, frame_period=frame_period)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)
    return f0s, timeaxes, sps, aps, coded_sps

def logf0_statistics(f0s):
    # Note: np.ma.log() calculating log on masked array (for incomplete or invalid entries in array)
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()
    return log_f0s_mean, log_f0s_std

def transpose_in_list(lst):
    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst

def coded_sps_normalization_fit_transform(coded_sps, coded_sps_mean=None, coded_sps_std=None):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    return coded_sps_concatenated
    # coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True) if coded_sps_mean is None else coded_sps_mean
    # coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True) if coded_sps_std is None else coded_sps_std
    # coded_sps_normalized = list()
    # for coded_sp in coded_sps:
    #     coded_sps_normalized.append(
    #         (coded_sp - coded_sps_mean) / coded_sps_std)
    # return coded_sps_normalized, coded_sps_mean, coded_sps_std

def wav_padding(wav, sr, frame_period, multiple=4):
    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) +
                                      1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right),
                        'constant', constant_values=0)
    return wav_padded

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):
    # Logarithm Gaussian Normalization for Pitch Conversions
    f0_converted = np.exp((np.ma.log(f0) - mean_log_src) /
                          std_log_src * std_log_target + mean_log_target).filled(0)
    return f0_converted

def world_decode_spectral_envelop(coded_sp, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
    return decoded_sp

def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    wav = wav.astype(np.float32)
    return wav
