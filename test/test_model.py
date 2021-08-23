import sys, os
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_vc_stft import GeneratorStft
from hparams import hparams as hp

def test_model():
    x = torch.ones(8, 2, 64, 513)
    e = torch.ones(8, 256)

    model = GeneratorStft(hp.dim_neck, hp.dim_emb, hp.dim_pre, hp.freq)

    o1, o2, c1 = model(x, e, e)
    c2 = model(x, e, None)

    assert tuple(o1.shape) == (8, 2, 64, 513), 'shapeはあっているか'
    assert tuple(o2.shape) == (8, 2, 64, 513), 'shapeはあっているか'
    assert tuple(c1.shape)  == (8, 2, 128), 'shapeはあっているか'
    assert tuple(c2.shape)  == (8, 2, 128), 'shapeはあっているか'
    assert o1.max() > 0, '正値を出すか'
    assert o1.max() > 0, '正値を出すか'
    assert o2.min() < 0, '負値を出すか'
    assert o2.min() < 0, '負値を出すか'