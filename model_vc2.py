import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        #torch.nn.init.xavier_uniform_(
        #    self.linear_layer.weight,
        #    gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        #torch.nn.init.xavier_uniform_(
        #    self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

    
class StyleEncoder(nn.Module):
    """Style Encoder module:
    """
    def __init__(self, dim_emb):
        super(StyleEncoder, self).__init__()
        
        self.lstm = nn.LSTM(80, 768, 2, batch_first=True)       
        self.linear = LinearNorm(768, dim_emb)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        
        emb = self.linear(outputs[:,-1,:])
        
        return emb

    
class ContentEncoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(ContentEncoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        linears = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(128+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
            linears.append(nn.Linear(256, 512))
        self.convolutions = nn.ModuleList(convolutions)
        self.linears = nn.ModuleList(linears)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2,1)
        c_org_se = c_org
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)

        for conv, linear in zip(self.convolutions, self.linears):
            x = F.relu(conv(x))
            x = x * torch.sigmoid(linear(c_org_se)).view(-1, 512, 1).expand_as(x)
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        

        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        # 0 torch.Size([8, 32]) torch.Size([8, 32])
        # 18 torch.Size([8, 32]) torch.Size([8, 32])
        return codes
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        linears = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
            linears.append(nn.Linear(256, dim_pre))
        self.convolutions = nn.ModuleList(convolutions)
        self.linears = nn.ModuleList(linears)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 128)

    def forward(self, x, c_trg):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv, linear in zip(self.convolutions, self.linears):
            x = F.relu(conv(x))
            x = x * torch.sigmoid(linear(c_trg)).view(-1, 512, 1).expand_as(x)
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(128, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 128,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(128))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)
        return x    
    

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        
        self.style_encoder = StyleEncoder(dim_emb)
        self.encoder = ContentEncoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x, c_org, c_trg):
        codes = self.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs, c_trg)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        #mel_outputs = mel_outputs.unsqueeze(1)
        #mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)


if __name__ == '__main__':
    x = torch.rand(8, 36, 128)
    e = torch.rand(8, 256)

    model = Generator(32, 256, 512, 18)

    out = model(x, e, e)

    for out in out:
        print(out.shape)
    
    

