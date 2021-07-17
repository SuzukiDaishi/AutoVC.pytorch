from numpy.lib.type_check import real
import torch
import torch.nn as nn
import torch.nn.functional as F

def split_complex(x, axis=1):
    real, imag = torch.chunk(x, 2, axis=axis)
    return real, imag

def cat_complex(xr, xi, axis=1):
    return torch.cat([xr, xi], axis=axis)

def apply_complex(fr, fi, xr, xi, axis=1):
    return cat_complex(fr(xr)-fi(xi), fr(xi)+fi(xr), axis=axis)

class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, complex_axis=1):
        super(ComplexLinear, self).__init__()
        mul = 1 # FIXME: なんか違う気がする
        self.fc_r = nn.Linear(in_features//mul, out_features//mul, bias=bias)
        self.fc_i = nn.Linear(in_features//mul, out_features//mul, bias=bias)
        self.complex_axis = complex_axis
    
    def forward(self, input):
        xr, xi = split_complex(input, axis=self.complex_axis)
        return apply_complex(self.fc_r, self.fc_i, xr, xi, axis=self.complex_axis)

class NaiveComplexBatchNorm1d(nn.Module):
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, complex_axis=1):
        super(NaiveComplexBatchNorm1d, self).__init__()
        mul = 2 if complex_axis == 1 else 1
        self.bn_r = nn.BatchNorm1d(num_features//mul, eps, momentum, affine, track_running_stats)
        self.bn_i = nn.BatchNorm1d(num_features//mul, eps, momentum, affine, track_running_stats)
        self.complex_axis = complex_axis

    def forward(self, input):
        xr, xi = split_complex(input, axis=self.complex_axis)
        return cat_complex(self.bn_r(xr), self.bn_i(xi), axis=self.complex_axis)

class ComplexConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, complex_axis=1):
        super(ComplexConv1d, self).__init__()
        mul = 2 if complex_axis == 1 else 1
        self.conv_r = nn.Conv1d(in_channels//mul, out_channels//mul, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv1d(in_channels//mul, out_channels//mul, kernel_size, stride, padding, dilation, groups, bias)
        self.complex_axis = complex_axis
        
    def forward(self, input):  
        xr, xi = split_complex(input, axis=self.complex_axis)
        return apply_complex(self.conv_r, self.conv_i, xr, xi, axis=self.complex_axis)

class NavieComplexLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False, complex_axis=2):
        super(NavieComplexLSTM, self).__init__()
        mul = 1 # FIXME: なんか違う気がする
        self.real_lstm = nn.LSTM(input_size//mul, hidden_size//mul, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.imag_lstm = nn.LSTM(input_size//mul, hidden_size//mul, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.complex_axis = complex_axis

    def forward(self, inputs):
        xr, xi = split_complex(inputs, axis=self.complex_axis) 
        r2r = self.real_lstm(xr)
        r2i = self.imag_lstm(xr)
        i2r = self.real_lstm(xi)
        i2i = self.imag_lstm(xi)
        return cat_complex(r2r[0]-i2i[0], i2r[0]+r2i[0], axis=self.complex_axis), [r2r[1], r2i[1], i2r[1], i2i[1]]

    def flatten_parameters(self):
        self.real_lstm.flatten_parameters()
        self.imag_lstm.flatten_parameters()

if __name__ == '__main__':
    x = torch.rand(8, 64, 80, dtype=torch.float32)

    lstm = nn.LSTM(80, 128, 2, batch_first=True)
    conv = ComplexConv1d(64, 128, 5, padding=2)
    
    o = lstm(x)[0]
    o = conv(o)
    print(o.shape)