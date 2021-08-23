import sys, os
import numpy as np
import simpleaudio as sa
import pysptk
from copy import deepcopy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dsp import load_wav
from utils.dsp import complex_stft, complex_istft
from hparams import hparams as hp

def play_audio(wave):
    wave = deepcopy(wave)
    wave *= 32767
    wave = wave.astype(np.int16)
    wave = sa.play_buffer(wave, 1, 2, hp.sample_rate)
    wave.wait_done()

def numpy_info(a, name=None, end='\n'):
    if name is not None:
        print(f'- - - {name} - - -')
    print('shape:', a.shape)
    print('dtype:', a.dtype)
    print('max:', a.max())
    print('min:', a.min())
    print('mean:', a.mean())
    print('', end=end)

def sample_stft_istft():
    path = pysptk.util.example_audio_file()
    wave = load_wav(path)
    
    numpy_info(wave, '変換前の音声')
    play_audio(wave)
    
    spec = complex_stft(wave)

    numpy_info(spec, 'スペクトログラム')

    out = complex_istft(spec, wave_length=wave.shape[0])
 
    numpy_info(out, '変換後の音声')
    play_audio(out)

if __name__ == '__main__':
    sample_stft_istft()