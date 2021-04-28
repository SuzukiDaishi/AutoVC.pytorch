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
import glob


class AudioDataloader():

    def __init__(self, batch_data_dir, batch_size=8):
        self.paths = sorted(glob.glob(f'{batch_data_dir}/batch_*.pkl'))
        self.batch_size = batch_size

    def __len__(self):
        return len(self.paths)

    def data_length(self):
        return len(self.paths) * self.batch_size
    
    def loader(self):
        for path in self.paths:
            mceps, emb = pickle.load(open(path, 'rb'))
            max_offsets = [ m.shape[1] - hp.n_frames + 1 for m in mceps ]
            sig_offsets = [ np.random.randint(0, offset) for offset in max_offsets]
            mceps = [ x[:, sig_offsets[i]:sig_offsets[i]+hp.n_frames] for i, x in enumerate(mceps) ]
            emb = torch.FloatTensor(emb)
            mceps = torch.FloatTensor(mceps)
            yield mceps, emb

if __name__ == '__main__':
    dl = AudioDataloader('data2')
    for m, e in dl.loader():
        print(m.shape, e.shape)
        # torch.Size([8, 36, 128]) torch.Size([8, 256])
