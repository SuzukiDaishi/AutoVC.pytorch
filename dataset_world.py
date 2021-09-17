import os
import math
import random
import json
import torch
from torch._C import wait
import torch.utils.data
import numpy as np
from spec_augment import spec_augment

from hparams import hparams as hp
from utils.dsp import load_wav
from utils.world import world_split, logsp_norm

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

def train_collate_world(batch):
    
    indexs = [ np.random.randint(0, b[0].shape[0] - (256*20)) for b in batch ]
    wavs = [ b[0][indexs[i]:(256*20-1)+indexs[i]] for i, b in enumerate(batch) ]
    wavs = [w * 2 ** (np.random.rand() * 2 - 1) for w in wavs]
    embs = [ b[1] for b in batch ]

    mels = np.array([ logsp_norm(np.log(world_split(w, use_ap=False)[-1])) for w in wavs ])
    embs = np.array(embs)

    mels = torch.from_numpy(mels.astype(np.float32)).clone()
    embs = torch.from_numpy(embs.astype(np.float32)).clone()

    return mels, embs

def test_collate_world(batch):
    wavs = [ b[0][:(256*20-1)] for b in batch ]
    embs = [ b[1] for b in batch ]

    del batch

    mels = np.array([ logsp_norm(np.log(world_split(w, use_ap=False)[-1])) for w in wavs ])
    embs = np.array(embs)

    mels = torch.from_numpy(mels.astype(np.float32)).clone()
    embs = torch.from_numpy(embs.astype(np.float32)).clone()

    return mels, embs
    

if __name__ == '__main__':
    data_path = './data'

    with open(os.path.join(data_path, 'train_data.json'), 'r') as f:
        train_data = json.load(f)

    with open(os.path.join(data_path, 'test_data.json'), 'r') as f:
        test_data = json.load(f)

    train_loader = torch.utils.data.DataLoader(
        AudiobookDataset(train_data),
        collate_fn=test_collate_world,
        batch_size=8, shuffle=True)
    
    max_ = -1000
    min_ = 1000
    len_ = len(train_loader)

    while True:
        for i, (m, e) in enumerate(train_loader):
            print(f'\n{i}/{len_}')
            print(m.shape, e.shape, m.dtype, e.dtype)
            print(m.max(), m.min(), m.mean())
            max_ = max(m.max(), max_)
            min_ = min(m.min(), min_)
            print(max_, min_)