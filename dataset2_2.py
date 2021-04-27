import os
import math
import random
import json
import torch
import torch.utils.data
import numpy as np
from hparams import hparams as hp
from utils import world, dsp
import pickle
import time

class AudiobookDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, train=False):
        self.data = input_data

    def __getitem__(self, index):
        p = self.data[index]
        fs = p['wav']
        e = p['emb']
        pw = p['world']
        
        f = random.choice(fs)
        wav = dsp.load_wav(f)
        emb = np.load(e)
        pw_data = np.load(pw)
        coded_sps_mean = pw_data['coded_sps_mean']
        coded_sps_std = pw_data['coded_sps_std']

        if len(wav) < hp.seq_len:
            wav = np.pad(wav, (0, hp.seq_len - len(wav)), mode='constant')
        
        return wav, emb, f, coded_sps_mean, coded_sps_std

    def __len__(self):
        return len(self.data)

def train_collate(batch):
    mel_win = hp.seq_len // hp.hop_length
    max_offsets = [x[0].shape[-1] - hp.seq_len + 1 for x in batch]

    # volume augmentation
    wavs = [w[0].astype(np.float64) * 2 ** (np.random.rand() * 2 - 1) for w in batch]

    mceps = world.world_encode_data(wavs, hp.sample_rate, frame_period=hp.frame_period, coded_dim=hp.num_mcep)[-1]
    
    # max_offsets = [ m.shape[0] - hp.n_frames + 1 for m in mceps ]
    # sig_offsets = [ np.random.randint(0, offset) for offset in max_offsets]

    # mceps = [ x[sig_offsets[i]:sig_offsets[i]+hp.n_frames] for i, x in enumerate(mceps) ]

    mceps = world.transpose_in_list(mceps)
    mceps = [ world.coded_sps_normalization_fit_transform([x], coded_sps_mean=batch[i][3], coded_sps_std=batch[i][4])[0][0] \
              for i, x in enumerate(mceps) ]

    emb = [x[1] for x in batch]
    fname = [x[2] for x in batch]

    # emb = torch.FloatTensor(emb)
    # mceps = torch.FloatTensor(mceps)

    return mceps, emb

def test_collate(batch):
    wavs = [world.wav_padding(w[0], hp.sample_rate, hp.frame_period).astype(np.float64) for w in batch]
    mceps = world.world_encode_data(wavs, hp.sample_rate, frame_period=hp.frame_period, coded_dim=hp.num_mcep)[-1]
    mceps = world.transpose_in_list(mceps)
    mceps = [ world.coded_sps_normalization_fit_transform([x], coded_sps_mean=batch[i][3], coded_sps_std=batch[i][4])[0] \
              for i, x in enumerate(mceps) ]
    embs = [x[1] for x in batch]
    emb = torch.FloatTensor(emb)
    mceps = torch.FloatTensor(mceps)
    return mceps, emb

if __name__ == '__main__':

    BASE_TRAIN_JSON = 'data/train_data.json'
    OUT_DIR = 'data2'
    BATCH_SIZE = 8

    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)

    with open(BASE_TRAIN_JSON, 'r') as f:
        train_data = json.load(f)
        train_loader = torch.utils.data.DataLoader(
                            AudiobookDataset(train_data),
                            collate_fn=train_collate,
                            batch_size=BATCH_SIZE, 
                            shuffle=True)
        t = None
        for batch_idx, (m, e) in enumerate(train_loader):
            num = f'{batch_idx+1}'.zfill(len(str(len(train_loader))))
            with open(f'{OUT_DIR}/batch_{num}.pkl', 'wb') as f:
                pickle.dump((m , e), f)
                print(f'{num}/{len(train_loader)}', time.time()-t if t is not None else 'start')
            t = time.time()