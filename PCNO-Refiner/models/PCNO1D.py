import torch.nn.functional as F
import torch
import torch.nn as nn
from utils import grid

from .condition_utils import ConditionedBlock, EmbedSequential, fourier_embedding

# ----------------------------------------------------------------------------------------------------------------------
# Baseline FNO: code from https://github.com/neural-operator/fourier_neural_operator
# ----------------------------------------------------------------------------------------------------------------------

################################################################
# Normalizer: code from https://github.com/zongyi-li/fourier_neural_operator
################################################################
# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

################################################################
# 1D fourier layer
################################################################
class FreqLinear(nn.Module):
    def __init__(self, in_channel, modes1):
        super().__init__()
        self.modes1 = modes1
        scale = 1 / (in_channel + 4 * modes1)
        self.weights = nn.Parameter(scale * torch.randn(in_channel, 4 * modes1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, 4 * modes1, dtype=torch.float32))

    def forward(self, x):
        B = x.shape[0]
        # print('x',x.shape)
        # print('self.weights',self.weights.shape)
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, 2, 2)
        return torch.view_as_complex(h)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        cond_channels = 256
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            # Number of Fourier modes to multiply, at most floor(N/2) + 1
            modes1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )
        self.cond_emb = FreqLinear(cond_channels, self.modes1)


    # Complex multiplication
    def compl_mul1d(self, input, weights, emb):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        # print('emb', emb.shape)
        # print('input', input.shape)
        temp = input * emb.unsqueeze(1)
        # print('temp', temp.shape)
        # print('weights', weights.shape)
        return torch.einsum("bix,iox->box", temp, weights)

    def forward(self, x, emb):
        # print('emb',emb.shape)
        emb12 = self.cond_emb(emb)
        emb1 = emb12[..., 0]
        emb2 = emb12[..., 1]
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1, emb1
        )
        out_ft[:, :, -self.modes1 :] = self.compl_mul1d(
            x_ft[:, :, -self.modes1 :], self.weights2, emb2
        )


        # Return to physical space
        return torch.fft.irfft(out_ft, n=x.size(-1))


class PCNO1d(nn.Module):
    def __init__(self, num_channels, hidden_channels, param_conditioning, modes=16, width=64, initial_step=10):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.param_conditioning = param_conditioning
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.activation=nn.GELU()
        self.fc0 = nn.Linear(
            initial_step * num_channels + 1, self.width
        )  # input channel is 2: (a(x), x)
        self.in_planes = hidden_channels

        time_embed_dim = hidden_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_channels, time_embed_dim),
            self.activation,
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.param_conditioning is not None:
            if self.param_conditioning.startswith("scalar"):
                num_params = 1 if "_" not in self.param_conditioning else int(self.param_conditioning.split("_")[1])
                self.pde_emb = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(hidden_channels, time_embed_dim),
                            self.activation,
                            nn.Linear(time_embed_dim, time_embed_dim),
                        )
                        for _ in range(num_params)
                    ]
                )
            else:
                raise NotImplementedError(f"Param conditioning {self.param_conditioning} not implemented")

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x, z=None):
        # x dim = [b, x1, t*v]
        # print('x_model0', x.shape)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        grid = self.get_grid(x.shape).to(x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        emb = 0
        if z is not None:
            if self.param_conditioning.startswith("scalar"):
                if z.ndim == 1:
                    z = z[:, None]
                for i in range(z.shape[-1]):
                    emb = emb + self.pde_emb[i](fourier_embedding(z[..., i], self.in_planes))
        # grid = self.get_grid(x.shape).to(x.device)
        # print('x_model1', x.shape)
        # x = torch.cat((x, grid), dim=-1)
        # print('x_model2',x.shape)
        # x = x.permute(0, 2, 1)
        # x = self.fc0(x)
        # x = x.permute(0, 2, 1)

        # pad the domain if input is non-periodic
        # x = F.pad(x, [0, self.padding])

        x1 = self.conv0(x, emb)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x, emb)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x, emb)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x, emb)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., : -self.padding]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        # print('x_final',x.shape)
        return x.unsqueeze(-2)

    # def get_grid(self, shape):
    #     batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    #     gridx = torch.linspace(0, 1, size_x)
    #     gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    #     gridy = torch.linspace(0, 1, size_y)
    #     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    #     return torch.cat((gridx, gridy), dim=-1)
    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        grid = torch.linspace(0, 1, size_x)
        grid = grid.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return grid
