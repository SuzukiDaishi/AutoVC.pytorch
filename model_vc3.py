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

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, num_speakers=256):
        super(Generator, self).__init__()
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
        self.residual_1 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_2 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_3 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_4 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_5 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_6 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_7 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_8 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_9 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)

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
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)

        x = x.contiguous().view(-1, 2304, width_size // 4)
        x = self.down_conversion(x)

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

        x = self.up_sample_1(x)
        x = self.up_sample_2(x)
        x = self.out(x)

        x = torch.squeeze(x, 1)

        return x, torch.flatten(codes, 1, 2)

if __name__ == '__main__':
    x = torch.rand(8, 36, 128)
    e = torch.rand(8, 256)

    model = Generator()

    out, c = model(x, e, e)

    print(out.shape, c.shape)
    