import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConditionalInstanceNormalisation(nn.Module):
    """AdaIN Block."""

    def __init__(self, dim_in, dim_c):
        super(ConditionalInstanceNormalisation, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dim_in = dim_in
        self.gamma_t = nn.Linear(dim_c, dim_in)
        self.beta_t = nn.Linear(dim_c, dim_in)

    def forward(self, x, c_trg):
        u = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-8)

        gamma = self.gamma_t(c_trg.to(self.device))
        gamma = gamma.view(-1, self.dim_in, 1)
        beta = self.beta_t(c_trg.to(self.device))
        beta = beta.view(-1, self.dim_in, 1)

        h = (x - u) / std
        h = h * gamma + beta

        return h

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, style_num):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.cin = ConditionalInstanceNormalisation(dim_out, style_num)
        self.glu = nn.GLU(dim=1)

    def forward(self, x, c):
        x = self.conv(x)
        x = self.cin(x, c)
        x = self.glu(x)

        return x

class SEResidualBlock(nn.Module):
    
    def __init__(self, dim_in, dim_out, style_num):
        super(SEResidualBlock, self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.cin = ConditionalInstanceNormalisation(dim_out, style_num)
        self.emb_se = nn.Linear(style_num, dim_out)
        self.glu = nn.GLU(dim=1)

    def forward(self, x, c):
        y = x
        y = self.conv(y)
        y = self.cin(y, c)
        y = y * torch.sigmoid(self.emb_se(c)).view(-1, 512, 1).expand_as(y)
        y = self.glu(y)
        y = y + x
        return y

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, num_speakers=256):
        super(Generator, self).__init__()

        self.emb_se_1 = nn.Linear(256, 64)
        self.emb_se_2 = nn.Linear(256, 128)
        self.emb_se_3 = nn.Linear(256, 256)
        self.emb_se_4 = nn.Linear(256, 256)

        self.emb_se_5 = nn.Linear(256, 256)
        self.emb_se_6 = nn.Linear(256, 128)
        self.emb_se_7 = nn.Linear(256, 64)

        # Down-sampling layers
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 9), padding=(1, 4), bias=False),
            nn.GLU(dim=1)
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )

        # Down-conversion layers.
        self.down_conversion = nn.Sequential(
            nn.Conv1d(in_channels=2304,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.InstanceNorm1d(num_features=256, affine=True)
        )

        # Bottleneck layers.
        self.residual_1 = SEResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_2 = SEResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_3 = SEResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_4 = SEResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_5 = SEResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_6 = SEResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_7 = SEResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_8 = SEResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_9 = SEResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)

        # Up-conversion layers.
        self.up_conversion = nn.Conv1d(in_channels=256,
                                       out_channels=2304,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)

        # Up-sampling layers.
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=128, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )

        # Out.
        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x, c_org, c_trg):
        x = torch.unsqueeze(x, 1)
        width_size = x.size(3)

        x = self.down_sample_1(x)
        x = x * torch.sigmoid(self.emb_se_1(c_org)).view(-1, 64, 1, 1).expand_as(x)
        x = self.down_sample_2(x)
        x = x * torch.sigmoid(self.emb_se_2(c_org)).view(-1, 128, 1, 1).expand_as(x)
        x = self.down_sample_3(x)
        x = x * torch.sigmoid(self.emb_se_3(c_org)).view(-1, 256, 1, 1).expand_as(x)

        x = x.contiguous().view(-1, 2304, width_size // 4)
        x = self.down_conversion(x)
        x = x * torch.sigmoid(self.emb_se_4(c_org)).view(-1, 256, 1).expand_as(x)

        codes = x

        if c_trg is None:
            return torch.flatten(codes, 1, 2)

        x = self.residual_1(x, c_trg)
        x = self.residual_2(x, c_trg)
        x = self.residual_3(x, c_trg)
        x = self.residual_4(x, c_trg)

        x = self.residual_5(x, c_trg)
        x = self.residual_6(x, c_trg)
        x = self.residual_7(x, c_trg)
        x = self.residual_8(x, c_trg)
        x = self.residual_9(x, c_trg)

        x = self.up_conversion(x)
        x = x.view(-1, 256, 9, width_size // 4)
        x = x * torch.sigmoid(self.emb_se_5(c_trg)).view(-1, 256, 1, 1).expand_as(x)

        x = self.up_sample_1(x)
        x = x * torch.sigmoid(self.emb_se_6(c_trg)).view(-1, 128, 1, 1).expand_as(x)
        x = self.up_sample_2(x)
        x = x * torch.sigmoid(self.emb_se_7(c_trg)).view(-1, 64, 1, 1).expand_as(x)
        x = self.out(x)

        x = torch.squeeze(x, 1)

        return x, torch.flatten(codes, 1, 2)

if __name__ == '__main__':
    x = torch.rand(8, 36, 128)
    e = torch.rand(8, 256)

    model = Generator()

    out, c = model(x, e, e)

    print(out.shape, c.shape)
    