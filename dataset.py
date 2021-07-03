import os
import math
import random
import json
import torch
import torch.utils.data
import numpy as np
from spec_augment import spec_augment

from hparams import hparams as hp
from utils.dsp import load_wav
from utils.dsp import melspectrogram, complex_melspectrogram

class AudiobookDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, train=False):
        self.data = input_data

    def __getitem__(self, index):
        p = self.data[index]
        fs = p['wav']
        e = p['emb']
        
        f = random.choice(fs)
        wav = load_wav(f)
        emb = np.load(e)

        if len(wav) < hp.seq_len:
            wav = np.pad(wav, (0, hp.seq_len - len(wav)), mode='constant')
           
        return wav, emb, f

    def __len__(self):
        return len(self.data)

def pad_seq(x, base=32):
    len_out = int(base * math.ceil(float(x.shape[1])/base))
    len_pad = len_out - x.shape[1]
    assert len_pad >= 0
    return np.pad(x, ((0,0), (0,len_pad)), 'constant'), len_pad

def train_collate(batch):
    mel_win = hp.seq_len // hp.hop_length
    
    max_offsets = [x[0].shape[-1] - hp.seq_len + 1 for x in batch]

    sig_offsets = [np.random.randint(0, offset) for offset in max_offsets]

    wav = [x[0][sig_offsets[i]:sig_offsets[i] + hp.seq_len] \
              for i, x in enumerate(batch)]
    
    # volume augmentation
    wav = [w * 2 ** (np.random.rand() * 2 - 1) for w in wav]
    
    mels = [melspectrogram(w[:-1]) for w in wav]
    
    # spec augmentation
    mels = [spec_augment(m) for m in mels]
    
    emb = [x[1] for x in batch]
    fname = [x[2] for x in batch]

    mels = torch.FloatTensor(mels)
    emb = torch.FloatTensor(emb)

    mels = mels.transpose(2,1)

    return mels, emb

def test_collate(batch):
    wavs = []
    embs = []
    for b in batch:
        wav = b[0]
        for p in range(0, len(wav), hp.seq_len):
            wav_seq = wav[p:p+hp.seq_len]
            if len(wav_seq) < hp.seq_len:
                wav_seq = np.pad(wav_seq, (0, hp.seq_len - len(wav_seq)), mode='constant')
            wavs.append(wav_seq)
            embs.append(b[1])
    
    mels = [pad_seq(melspectrogram(w))[0] for w in wavs]

    mels = torch.FloatTensor(mels)
    embs = torch.FloatTensor(embs)
    
    mels = mels.transpose(2,1)
    
    return mels, embs

def complex_train_collate(batch):
    mel_win = hp.seq_len // hp.hop_length
    max_offsets = [x[0].shape[-1] - hp.seq_len + 1 for x in batch]
    sig_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    wav = [x[0][sig_offsets[i]:sig_offsets[i] + hp.seq_len] \
              for i, x in enumerate(batch)]
    wav = [w * 2 ** (np.random.rand() * 2 - 1) for w in wav]
    mels = [complex_melspectrogram(w[:-1]) for w in wav]
    mels_real = [spec_augment(m.real) for m in mels]
    mels_imag = [spec_augment(m.imag) for m in mels]
    mels = [ mr + 1j * mi for mr, mi in zip(mels_real, mels_imag) ]

    emb = [x[1] + 1j * x[1] for x in batch]
    fname = [x[2] for x in batch]

    mels = torch.tensor(mels, dtype=torch.cfloat)
    embs = torch.tensor(emb, dtype=torch.cfloat)

    mels = mels.transpose(2,1)

    return mels, embs

def complex_test_collate(batch):
    wavs = []
    embs = []
    for b in batch:
        wav = b[0]
        for p in range(0, len(wav), hp.seq_len):
            wav_seq = wav[p:p+hp.seq_len]
            if len(wav_seq) < hp.seq_len:
                wav_seq = np.pad(wav_seq, (0, hp.seq_len - len(wav_seq)), mode='constant')
            wavs.append(wav_seq)
            embs.append(b[1])
    
    mels = [ pad_seq(complex_melspectrogram(w))[0] for w in wavs ]
    embs = [ e + 1j * e for e in embs ]

    mels = torch.tensor(mels, dtype=torch.cfloat)
    embs = torch.tensor(embs, dtype=torch.cfloat)
    
    mels = mels.transpose(2,1)
    
    return mels, embs

if __name__ == '__main__':

    data_path = './data'

    with open(os.path.join(data_path, 'train_data.json'), 'r') as f:
        train_data = json.load(f)

    with open(os.path.join(data_path, 'test_data.json'), 'r') as f:
        test_data = json.load(f)

    train_loader = torch.utils.data.DataLoader(
        AudiobookDataset(train_data),
        collate_fn=complex_train_collate,
        batch_size=8, shuffle=True)
    
    for m, e in train_loader:
        print(m.shape, e.shape, m.dtype, e.dtype)