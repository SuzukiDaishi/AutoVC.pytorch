import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dsp import load_wav, stft, istft
from utils.dsp import split_complex, join_complex
from utils.dsp import stft_normalize, stft_unnormalize
from utils.dsp import complex_stft, complex_istft


# AutoVCのもともとはあっていると仮定

path = './VCTK-Corpus/wav48/p225/p225_001.wav'
x = load_wav(path)
spec = stft(x)

def test_stft():
    '''
    stft関数の検証
    '''
    y = istft(spec, wave_length=x.shape[0])

    assert x.shape == y.shape, 'shapeを復元できてない'
    np.testing.assert_almost_equal(x, y, decimal=7, err_msg='ほとんど復元できない')

def test_join_complex():
    '''
    join_complex関数の検証
    '''
    _spec = split_complex(spec)
    y_spec = join_complex(_spec)
    
    assert spec.shape == y_spec.shape, 'shapeを復元できてない'
    assert spec.dtype == y_spec.dtype, '型が変更してしまっている'
    np.testing.assert_equal(spec, y_spec, err_msg='値が復元できていない')

def test_stft_normalize():
    '''
    stft_normalize関数の検証
    '''
    _spec = split_complex(spec)
    y_spec = stft_normalize(_spec)
    
    assert _spec.shape == y_spec.shape, 'shapeを復元できてない'
    assert y_spec.dtype == np.float32, '型が変更してしまっている'
    # assert y_spec.max() <= 1, '1より大きい数字があります'
    # assert y_spec.min() >= 0, '0より小さい数字があります'

def test_stft_unnormalize():
    '''
    stft_unnormalize関数の検証
    '''
    x_spec = split_complex(spec)
    _spec = stft_normalize(x_spec)
    y_spec = stft_unnormalize(_spec)

    assert x_spec.shape == y_spec.shape, 'shapeを復元できてない'
    assert x_spec.dtype == y_spec.dtype, '型が変更してしまっている'
    np.testing.assert_almost_equal(x_spec, y_spec, decimal=4, err_msg='ほとんど復元できない')

def test_complex_stft_istft():
    '''
    一通りの流れの検証
    '''
    y = complex_stft(x)
    y = complex_istft(y, wave_length=x.shape[0])

    assert x.shape == y.shape, 'shapeを復元できてない'
    assert x.dtype == y.dtype, '型が変更してしまっている'
    np.testing.assert_almost_equal(x, y, decimal=5, err_msg='ほとんど復元できない')
