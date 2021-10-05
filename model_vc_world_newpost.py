import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_vc_world import StyleEncoder, ContentEncoder, Decoder, Postnet

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1, 1], stride=[1, 1], padding='same', dilation=[1, 1], bias=True, groups=1):
        super(ConvNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                    padding=padding, dilation=dilation, bias=bias, groups=groups)
    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.Mish(inplace=True),
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=[3, 3]):
        super(ResBlock, self).__init__()
        self.conv = ConvNorm(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, signal):
        res = signal
        out = self.conv(signal)
        out = self.bn(out)
        out += res
        return out

class SEResBlock(nn.Module):

    def __init__(self, channels):
        super(SEResBlock, self).__init__()
        self.conv = ConvNorm(channels, channels, kernel_size=[3, 3], groups=channels)
        self.bn = nn.BatchNorm2d(channels)
        self.se = SELayer(channels)
        
    def forward(self, signal):
        res = signal
        out = self.conv(signal)
        out = self.bn(out)
        out = self.se(out)
        out += res
        return out

class Postnet2D(nn.Module):
    """Postnet 2D"""

    def __init__(self):
        super(Postnet2D, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(1, 8, kernel_size=[15,5], stride=1),
                nn.BatchNorm2d(8)
            )
        )
        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(8, 8, kernel_size=[15,5], stride=1),
                    nn.BatchNorm2d(8)
                )
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(8, 1, kernel_size=[15,5], stride=1)
            )
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
        x = self.convolutions[-1](x)
        x = x.squeeze(1)
        return x

class PostnetSE(nn.Module):
    """Postnet SE"""

    def __init__(self):
        super(PostnetSE, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            ResBlock(1, 8, kernel_size=[5, 3])
        )
        for i in range(1, 5 - 1):
            self.convolutions.append(
                SEResBlock(8)
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(8, 1)
            )
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
        x = self.convolutions[-1](x)
        x = x.squeeze(1)
        return x
    
if __name__ == '__main__':
    model1 = Postnet()
    model2 = Postnet2D()
    model3 = PostnetSE()

    x = torch.rand(8, 513, 64)

    print('POSTNET 1D')
    print('input:', x.shape)

    t1 = time.time()
    o1 = model1(x)

    print('output:', o1.shape, time.time() - t1)


    print('POSTNET 2D')
    print('input:', x.shape)

    t1 = time.time()
    o2 = model2(x)

    print('output:', o2.shape, time.time() - t1)

    print('POSTNET SE')
    print('input:', x.shape)

    t1 = time.time()
    o3 = model3(x)

    print('output:', o3.shape, time.time() - t1)

    torch.onnx.export(model1, x, './postnet_1.onnx', verbose=True,input_names=['input'],output_names=['output'])
    torch.onnx.export(model2, x, './postnet_2.onnx', verbose=True,input_names=['input'],output_names=['output'])
    torch.onnx.export(model3, x, './postnet_3.onnx', verbose=True,input_names=['input'],output_names=['output'])




