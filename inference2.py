import argparse
import math
import json
import numpy as np
import os
import torch
import librosa
from hparams import hparams as hp
from utils import world
from utils.dsp import load_wav
from model_vc2 import Generator
from synthesis import build_model
from synthesis import wavegen

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output', type=str, required=True, help='path to output wav(./output.wav)')
    parser.add_argument('--src-wav', type=str, required=True, help='path to src wav(./data/test/[speaker]/[filename]/0000.wav')
    parser.add_argument('--src-emb', type=str, required=True, help='path to src wav(./data/test/[speaker]/emb.npy')
    parser.add_argument('--tgt-emb', type=str, required=True, help='path to src wav(./data/test/[speaker]/emb.npy')
    parser.add_argument('--src-world', type=str, required=True, help='data/test/[speaker]/world.npz')
    parser.add_argument('--tgt-world', type=str, required=True, help='data/test/[speaker]/world.npz')
    parser.add_argument('--vocoder', type=str, required=True, help='path to checkpoint_step001000000_ema.pth')
    parser.add_argument('--autovc', type=str, required=True, help='checkpoints/checkpoint_step000600.pth')
    args = parser.parse_args()
    
    output_path = args.output
    src_wav_path = args.src_wav
    src_emb_path = args.src_emb
    tgt_emb_path = args.tgt_emb
    src_world_path = args.src_world
    tgt_world_path = args.tgt_world
    vocoder_checkpoint_path = args.vocoder
    autovc_checkpoint_path = args.autovc

    dim_neck = 32
    dim_emb = 256
    dim_pre = 512
    freq = 18
    sample_rate = 16000 
    coded_dim = 36

    device = torch.device('cpu')
    wavnet = build_model().to(device)
    checkpoint = torch.load(vocoder_checkpoint_path, map_location=device)
    wavnet.load_state_dict(checkpoint["state_dict"])

    wav = load_wav(src_wav_path).astype(np.float64)
    emb = np.load(src_emb_path)
    emb_tgt = np.load(tgt_emb_path)
    pw_data = np.load(src_world_path)
    coded_sps_mean = pw_data['coded_sps_mean']
    coded_sps_std = pw_data['coded_sps_std']
    log_f0_mean = pw_data['log_f0_mean']
    log_f0_std = pw_data['log_f0_std']

    pw_data_tgt = np.load(tgt_world_path)
    coded_sps_mean_tgt = pw_data_tgt['coded_sps_mean']
    coded_sps_std_tgt = pw_data_tgt['coded_sps_std']
    log_f0_mean_tgt = pw_data_tgt['log_f0_mean']
    log_f0_std_tgt = pw_data_tgt['log_f0_std']

    frame_period=5.0
    f0, timeaxis, sp, ap = world.world_decompose(wav, sample_rate, frame_period=5.0)
    coded_sp = world.world_encode_spectral_envelop(sp=sp, fs=sample_rate, dim=coded_dim)
    pad_size = (128*((coded_sp.shape[0]//128)+1)) - coded_sp.shape[0]
    coded_sps = np.pad(coded_sp, ((0, pad_size), (0, 0)), 'constant').reshape(-1, 128, coded_dim)
    mceps = world.transpose_in_list(coded_sps)
    mceps = world.coded_sps_normalization_fit_transform(mceps, coded_sps_mean=coded_sps_mean, coded_sps_std=coded_sps_std)[0]
    mceps = torch.FloatTensor(mceps)

    emb = [emb for _ in range(mceps.shape[0])]
    emb = torch.FloatTensor(emb)
    emb_tgt = [emb_tgt for _ in range(mceps.shape[0])]
    emb_tgt = torch.FloatTensor(emb_tgt)

    model = Generator(dim_neck, dim_emb, dim_pre, freq)

    checkpoint = torch.load(autovc_checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    x = mceps
    e = emb
    et = emb_tgt

    mel_otputs, mel_outputs_postnet, codes = model(x, e, et)
    out = (mel_outputs_postnet.cpu().detach().numpy() * coded_sps_std_tgt + coded_sps_mean_tgt).transpose(0,2,1).reshape(-1, 36)[:-pad_size]
    out = world.world_decode_spectral_envelop(out, sample_rate)
    out_f0 = world.pitch_conversion(f0, log_f0_mean, log_f0_std, log_f0_mean_tgt, log_f0_std_tgt)
    waveform = world.world_speech_synthesis(out_f0, out, ap, sample_rate, frame_period)
    librosa.output.write_wav(output_path, waveform, sr=16000)