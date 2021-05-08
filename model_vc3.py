import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    def __init__(self, jitter_rate=0):
        super(Generator, self).__init__()
        self.encoder = Encoder(jitter_rate)
        self.decoder = Decoder()

    def forward(self, x, src_m, tgt_m=None):
        z, loss, perplexity = self.encoder(x, src_m)
        if tgt_m is None :
            return z, loss, perplexity
        else :
            y = self.decoder(z, tgt_m)
            return y, z, loss, perplexity
    

    def convert(self, x, src_m, tgt_m):
        z, indices = self.encoder.encode(x, src_m)
        y = self.decoder(z, tgt_m)
        return y, indices


class Encoder(nn.Module):
    def __init__(self, jitter_rate=0):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 256)

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=128,
                               kernel_size=(5, 15),
                               stride=(1, 1),
                               padding=(2, 7))
        
        self.conv1_gates = nn.Conv2d(in_channels=1,
                                     out_channels=128,
                                     kernel_size=(5, 15),
                                     stride=1,
                                     padding=(2, 7))
        
        self.downSample1 = downSample_Generator(in_channels=128,
                                                out_channels=256,
                                                kernel_size=5,
                                                stride=2,
                                                padding=2)
        
        self.downSample2 = downSample_Generator(in_channels=256,
                                                out_channels=256,
                                                kernel_size=5,
                                                stride=2,
                                                padding=2)
        
        self.conv2dto1dLayer = nn.Sequential(nn.Conv1d(in_channels=2304,
                                                       out_channels=256,
                                                       kernel_size=1,
                                                       stride=1,
                                                       padding=0),
                                             nn.InstanceNorm1d(num_features=256,
                                                               affine=True))
        
        self.residualLayer1 = ResidualLayer(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)

        self.residualLayer2 = ResidualLayer(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        
        self.residualLayer3 = ResidualLayer(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        
        self.codebook = VQEmbeddingEMA(512, 64)
        self.jitter = Jitter(jitter_rate)

    def forward(self, x, e):
        x = x.unsqueeze(1)
        y = self.conv1(x) * torch.sigmoid(self.conv1_gates(x))
        y = y * torch.sigmoid(self.linear1(e).view(-1, 128, 1, 1).expand_as(y))
        y = self.downSample1(y)
        y = y * torch.sigmoid(self.linear2(e).view(-1, 256, 1, 1).expand_as(y))
        y = self.downSample2(y)
        y = y * torch.sigmoid(self.linear3(e).view(-1, 256, 1, 1).expand_as(y))
        y = y.view(y.size(0), 2304, 1, -1)
        y = y.squeeze(2)
        y = self.conv2dto1dLayer(y)
        y = y * torch.sigmoid(self.linear4(e).view(-1, 256, 1).expand_as(y))
        y = self.residualLayer1(y)
        y = self.residualLayer2(y)
        y = self.residualLayer3(y)
        y = self.conv2(y.transpose(1, 2))
        z, loss, perplexity = self.codebook(y)
        z = self.jitter(z)
        return z, loss, perplexity
    
    def encode(self, x, e):
        x = x.unsqueeze(1)
        y = self.conv1(x) * torch.sigmoid(self.conv1_gates(x))
        y = y * torch.sigmoid(self.linear1(e).view(-1, 128, 1, 1).expand_as(y))
        y = self.downSample1(y)
        y = y * torch.sigmoid(self.linear2(e).view(-1, 256, 1, 1).expand_as(y))
        y = self.downSample2(y)
        y = y * torch.sigmoid(self.linear3(e).view(-1, 256, 1, 1).expand_as(y))
        y = y.view(y.size(0), 2304, 1, -1)
        y = y.squeeze(2)
        y = self.conv2dto1dLayer(y)
        y = y * torch.sigmoid(self.linear4(e).view(-1, 256, 1).expand_as(y))
        y = self.residualLayer1(y)
        y = self.residualLayer2(y)
        y = self.residualLayer3(y)
        z, indices = self.codebook.encode(z.transpose(1, 2))
        return z, indices


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 256)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1, stride=1)

        self.residualLayer4 = ResidualLayer(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)

        self.residualLayer5 = ResidualLayer(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
                                            
        self.residualLayer6 = ResidualLayer(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)

        self.conv1dto2dLayer = nn.Sequential(nn.Conv1d(in_channels=256,
                                                       out_channels=2304,
                                                       kernel_size=1,
                                                       stride=1,
                                                       padding=0),
                                             nn.InstanceNorm1d(num_features=2304,
                                                               affine=True))

        self.upSample1 = self.upSample(in_channels=256,
                                       out_channels=1024,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)
        
        self.upSample2 = self.upSample(in_channels=256,
                                       out_channels=512,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)
        
        self.lastConvLayer = nn.Conv2d(in_channels=128,
                                       out_channels=1,
                                       kernel_size=(5, 15),
                                       stride=(1, 1),
                                       padding=(2, 7))

    def forward(self, x, e):
        y = self.conv1(x).transpose(1, 2)
        y = y * torch.sigmoid(self.linear1(e).view(-1, 256, 1).expand_as(y))
        y = self.residualLayer4(y)
        y = y * torch.sigmoid(self.linear2(e).view(-1, 256, 1).expand_as(y))
        y = self.residualLayer5(y)
        y = y * torch.sigmoid(self.linear3(e).view(-1, 256, 1).expand_as(y))
        y = self.residualLayer6(y)
        y = y * torch.sigmoid(self.linear4(e).view(-1, 256, 1).expand_as(y))
        y = self.conv1dto2dLayer(y)
        y = y.unsqueeze(2)
        y = y.view(y.size(0), 256, 9, -1)
        y = self.upSample1(y)
        y = self.upSample2(y)
        y = self.lastConvLayer(y).squeeze(1)
        return y 

    def upSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.PixelShuffle(upscale_factor=2),
                                       nn.InstanceNorm2d(
                                           num_features=out_channels // 4,
                                           affine=True),
                                       GLU())
        return self.convLayer



class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_count', torch.zeros(n_embeddings))
        self.register_buffer('ema_weight', self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

class Jitter(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        prob = torch.Tensor([p / 2, 1 - p, p / 2])
        self.register_buffer('prob', prob)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        else:
            batch_size, sample_size, channels = x.size()

            dist = Categorical(self.prob)
            index = dist.sample(torch.Size([batch_size, sample_size])) - 1
            index[:, 0].clamp_(0, 1)
            index[:, -1].clamp_(-1, 0)
            index += torch.arange(sample_size, device=x.device)

            x = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, channels))
        return x

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()

        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=padding),
                                          nn.InstanceNorm1d(num_features=out_channels,
                                                            affine=True))

        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=out_channels,
                                                                affine=True))

        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=in_channels,
                                                                affine=True))
    
    def forward(self, x):
        h1_norm = self.conv1d_layer(x)
        h1_gates_norm = self.conv_layer_gates(x)
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)
        h2_norm = self.conv1d_out_layer(h1_glu)
        return x + h2_norm

class downSample_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(downSample_Generator, self).__init__()
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         affine=True))
        self.convLayer_gates = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding),
                                             nn.InstanceNorm2d(num_features=out_channels,
                                                               affine=True))
    def forward(self, x):
        return self.convLayer(x) * torch.sigmoid(self.convLayer_gates(x))


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

if __name__ == '__main__':
    m = torch.rand(8, 36, 128)
    e = torch.rand(8, 256)

    model = Generator()

    z, loss, perplexity = model(m, e, e)
    print(z.shape, loss, perplexity)
