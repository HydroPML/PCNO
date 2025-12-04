import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from utils import grid, fc
from .spectral import (
    fft_expand_dims,
    fft_mesh_2d,
    spectral_div_2d,
    spectral_grad_2d,
    spectral_laplacian_2d,
)

from .spectral import (
    fft_expand_dims3d,
    fft_mesh_3d,
    spectral_div_3d,
    spectral_grad_3d,
    spectral_laplacian_3d,
)
from einops import repeat
###########################
# Invirance
###########################
class GConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, first_layer=False, last_layer=False,
                 spectral=False, Hermitian=False, reflection=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)
        assert kernel_size % 2 == 1, "kernel size must be odd"
        dtype = torch.cfloat if spectral else torch.float
        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Hermitian else kernel_size
        self.Hermitian = Hermitian
        if first_layer or last_layer:
            self.W = nn.Parameter(torch.empty(out_channels, 1, in_channels, self.kernel_size_Y, self.kernel_size_X, dtype=dtype))
        else:
            if self.Hermitian:
                self.W = nn.ParameterDict({
                    'y0_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_X - 1, 1, dtype=dtype)),
                    'yposx_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y, self.kernel_size_X - 1, dtype=dtype)),
                    '00_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, 1, 1, dtype=torch.float))
                })
            else:
                self.W = nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y, self.kernel_size_X, dtype=dtype))
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1)) if bias else None
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        if self.Hermitian:
            for v in self.W.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):

        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        if self.Hermitian:
            self.weights = torch.cat([self.W["y0_modes"], self.W["00_modes"].cfloat(), self.W["y0_modes"].conj()], dim=-2)
            self.weights = torch.cat([self.weights, self.W["yposx_modes"]], dim=-1)
            self.weights = torch.cat([self.weights[..., 1:].conj().rot90(k=2, dims=[-2, -1]), self.weights], dim=-1)
        else:
            self.weights = self.W[:]

        if self.first_layer or self.last_layer:

            # construct the weight
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1)

            # apply each of the group elements to the corresponding repetition
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k].rot90(k=k, dims=[-2, -1])

            # apply each the reflection group element to the rotated kernels
            if self.reflection:
                self.weights[:, self.rt_group_size:] = self.weights[:, :self.rt_group_size].flip(dims=[-2])

            # collapse out_channels and group1 dimensions for use with conv2d
            if self.first_layer:
                self.weights = self.weights.view(-1, self.in_channels, self.kernel_size_Y, self.kernel_size_Y)
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
            else:
                self.weights = self.weights.transpose(2, 1).reshape(self.out_channels, -1, self.kernel_size_Y, self.kernel_size_Y)
                self.bias = self.B

        else:

            # construct the weight
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1)

            # apply elements in the rotation group
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k - 1].rot90(dims=[-2, -1])

                if self.reflection:
                    self.weights[:, k] = torch.cat([self.weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                                                    self.weights[:, k, :, :(self.rt_group_size - 1)],
                                                    self.weights[:, k, :, (self.rt_group_size + 1):],
                                                    self.weights[:, k, :, self.rt_group_size].unsqueeze(2)], dim=2)
                else:
                    self.weights[:, k] = torch.cat([self.weights[:, k, :, -1].unsqueeze(2), self.weights[:, k, :, :-1]], dim=2)

            if self.reflection:
                # apply elements in the reflection group
                self.weights[:, self.rt_group_size:] = torch.cat(
                    [self.weights[:, :self.rt_group_size, :, self.rt_group_size:],
                     self.weights[:, :self.rt_group_size, :, :self.rt_group_size]], dim=3).flip([-2])

            # collapse out_channels / groups1 and in_channels/groups2 dimensions for use with conv2d
            self.weights = self.weights.view(self.out_channels * self.group_size, self.in_channels * self.group_size,
                                             self.kernel_size_Y, self.kernel_size_Y)
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)

        if self.Hermitian:
            self.weights = self.weights[..., -self.kernel_size_X:]

    def forward(self, x):

        self.get_weight()

        # output is of shape (batch * out_channels, number of group elements, ny, nx)
        x = nn.functional.conv2d(input=x, weight=self.weights)

        # add the bias
        if self.B is not None:
            x = x + self.bias
        return x

class GSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, reflection=False):
        super(GSpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.conv = GConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * modes - 1,
                            reflection=reflection, bias=False, spectral=True, Hermitian=True)
        self.get_weight()

    # Building the weight
    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        self.get_weight()

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes), :self.modes]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.weights.shape[0], x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes), :self.modes] = \
            self.compl_mul2d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1)))

        return x


class GNorm(nn.Module):
    def __init__(self, width, group_size):
        super().__init__()
        self.group_size = group_size
        self.norm = torch.nn.InstanceNorm3d(width)

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.group_size, x.shape[-2], x.shape[-1])
        x = self.norm(x)
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        return x


class GMLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, modes, reflection=False, last_layer=False):
        super(GMLP2d, self).__init__()
        # self.mlp1 = GConv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, reflection=reflection)
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.modes = modes
        # self.mlp2 = GConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, reflection=reflection,
        #                     last_layer=last_layer)
        self.conv = GSpectralConv2d(in_channels=in_channels, out_channels=in_channels, modes=self.modes, reflection=reflection)
        self.mlp2 = GConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, reflection=reflection, last_layer=last_layer)
        self.w = GConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, reflection=reflection, last_layer=last_layer)
        self.gnorm = GNorm(in_channels, group_size=4 * (1 + reflection))

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x1 = self.gnorm(self.conv(self.gnorm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w(x)
        # print('x1', x1.shape)
        # print('x2', x2.shape)
        x = x1 + x2
        # x = self.mlp2(x)
        return x


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
# 2D fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2d, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class MLP2d2(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, modes):
        super(MLP2d2, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.conv = SpectralConv2d(mid_channels, mid_channels, self.modes1, self.modes2)
        self.norm = nn.InstanceNorm2d(mid_channels)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.norm(self.conv(self.norm(x)))
        x = self.mlp2(x)
        return x


class HelmholtzProjection(nn.Module):
    def __init__(
            self,
            n_grid: int = 64,
            diam: float = 2 * torch.pi,
            dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        """
        Perform Helmholtz decomposition in the frequency domain
        to project any vector field to divergence-free.

        Modified for 4D input where uhat has shape (bsz, 2, nx, ny)
        """
        self.n_grid = n_grid
        self.diam = diam
        # self._update_fft_mesh(n_grid, diam, dtype)

    def _update_fft_mesh(self, nx, ny, diam=None, dtype=torch.float32):
        diam = diam if diam is not None else self.diam
        kx, ky = fft_mesh_2d(nx, ny, diam)
        lap = spectral_laplacian_2d(fft_mesh=(kx, ky))
        self.register_buffer("lap", lap.to(dtype))
        self.register_buffer("kx", kx.to(dtype))
        self.register_buffer("ky", ky.to(dtype))

    @staticmethod
    def div(uhat, fft_mesh):
        """
        uhat: (b, 2, nx, ny)
        Returns: divergence of uhat
        """
        kx, ky = fft_expand_dims(fft_mesh, uhat.size(0))
        return spectral_div_2d([uhat[:, 0], uhat[:, 1]], (kx, ky))

    @staticmethod
    def grad(uhat, fft_mesh):
        """
        uhat: (b, nx, ny)
        Returns: gradient of uhat as (b, 2, nx, ny)
        """
        kx, ky = fft_expand_dims(fft_mesh, uhat.size(0))
        graduhat = spectral_grad_2d(uhat, (kx, ky))
        return torch.stack(graduhat, dim=1)

    def forward(self, uhat):
        """
        uhat: (b, 2, nx, ny) - 4D input where the 2nd dimension is velocity components
        """
        bsz, _, nx, ny = uhat.shape

        # Update FFT mesh if the grid size changes (evaluation mode)
        # if nx != self.n_grid:
        self._update_fft_mesh(nx, ny)
        fft_mesh = (self.kx, self.ky)

        # Calculate divergence
        div_u = self.div(uhat, fft_mesh)

        # Calculate the gradient of the divergence
        grad_div_u = self.grad(div_u, fft_mesh)

        # Apply the Laplacian operator
        lap = repeat(self.lap, "x y -> b 2 x y", b=bsz)  # Repeat lap for the batch size
        # lap = lap[..., :uhat.shape[-1]]

        # Project the input field to make it divergence-free
        print('uhat',uhat.shape)
        print('grad_div_u', grad_div_u.shape)
        print('lap', lap.shape)
        w_hat = uhat - grad_div_u / lap
        return w_hat

class SpectralConv2d_O(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_O, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.postprocess = HelmholtzProjection(n_grid=64)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1), dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        # print('out_ft',out_ft.shape)

        out_ft = self.postprocess(out_ft)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class PCNO2d(nn.Module):
    def __init__(self, num_channels, modes1, modes2, width, initial_step, grid_type):
        super(PCNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8  # pad the domain if input is non-periodic
        self.fc_d, self.fc_C, self.fc_filter = 3, 25, False
        self.fc_pad = fc(fc_d=self.fc_d, fc_C=self.fc_C)
        reflection = False

        self.grid = grid(twoD=True, grid_type=grid_type)
        self.p = nn.Linear(initial_step * num_channels + self.grid.grid_dim, self.width)  # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv3 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes1,
        #                              reflection=reflection)
        self.mlp0 = MLP2d(self.width, self.width, self.width)
        self.mlp1 = MLP2d(self.width, self.width, self.width)
        self.mlp2 = MLP2d(self.width, self.width, self.width)
        self.mlp3 = MLP2d(self.width, self.width, self.width)
        # self.mlp3 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width,
        #                    reflection=reflection)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        # self.qm = nn.Conv2d(self.width, self.width * 4, 1)
        # self.q = MLP2d(self.width, num_channels, self.width * 4)  # output channel is 1: u(x, y)
        # self.w3 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=1, reflection=reflection)
        # self.gnorm = GNorm(self.width, group_size=4 * (1 + reflection))
        self.q = GMLP2d(in_channels=self.width, out_channels=num_channels, mid_channels=self.width * 4, modes=self.modes1,
                        reflection=reflection, last_layer=True)
        # self.q = MLP2d2(self.width, num_channels, self.width * 4, self.modes1)
        # self.mc = SpectralConv2d_O(num_channels, num_channels, self.modes1, self.modes2)

        # self.mc = nn.Linear(num_channels, num_channels-1)


    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        grid = self.get_grid(x.shape).to(x.device)
        x = torch.cat((x, grid), dim=-1)
        # x = self.grid(x)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        # x = self.qm(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # x = F.gelu(x)
        # x1 = self.gnorm(self.conv3(self.gnorm(x)))
        # x1 = self.mlp3(x1)
        # x2 = self.w3(x)
        # x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        # x = x.permute(0, 2, 3, 1)
        # print('x', x.shape)
        # x = self.mc(x)
        # x = x.permute(0, 3, 1, 2)
        x = x.permute(0, 2, 3, 1)
        # print('x',x.shape)
        return x.unsqueeze(-2)
        # x = self.mc(x)

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.arange(0, 2.0 * torch.pi, 2.0 * torch.pi / size_x)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(1.875, 178.125, size_y) / 180 * torch.pi
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

    # def sample(self, x, ts: List[float], x_self_cond, z):
    #     # print('torch.tensor(ts[0]).cuda()', torch.tensor(ts[0]).shape)
    #     x = self(x, torch.tensor(ts[0]).cuda().unsqueeze(0), x_self_cond, z)
    #
    #
    #     for t in ts[1:]:
    #         t = torch.tensor(t).cuda()
    #         t = t.unsqueeze(0)
    #         zz = torch.randn_like(x)
    #         x = x + math.sqrt(t**2 - self.eps**2) * zz
    #         x = self(x, t, x_self_cond, z)
    #
    #     return x


################################################################
# 3d fourier layers
################################################################
class GConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_size_T, bias=True, first_layer=False, last_layer=False,
                 spectral=False, Hermitian=False, reflection=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)
        assert kernel_size % 2 == 1, "kernel size must be odd"
        dtype = torch.cfloat if spectral else torch.float
        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Hermitian else kernel_size
        self.kernel_size_T_full = kernel_size_T
        self.kernel_size_T = kernel_size_T // 2 + 1 if Hermitian else kernel_size_T
        self.Hermitian = Hermitian
        if first_layer or last_layer:
            self.W = nn.Parameter(torch.empty(out_channels, 1, in_channels, self.kernel_size_Y, self.kernel_size_X,
                                  self.kernel_size_T, dtype=dtype))
        else:
            if self.Hermitian:
                self.W = nn.ParameterDict({
                    'y00_modes':torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size,
                                                               self.kernel_size_X - 1, 1, 1, dtype=torch.cfloat)),
                    'yposx0_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size,
                                                                   self.kernel_size_Y, self.kernel_size_X - 1, 1,
                                                                   dtype=torch.cfloat)),
                    '000_modes':torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, 1, 1, 1)),
                    'yxpost_modes':torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size,
                                                                  self.kernel_size_Y, self.kernel_size_Y,
                                                                  self.kernel_size_T - 1, dtype=torch.cfloat))
                })
            else:
                self.W = nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y,
                                                  self.kernel_size_X, self.kernel_size_T, dtype=dtype))
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1, 1)) if bias else None
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        if self.Hermitian:
            for v in self.W.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):

        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        if self.Hermitian:
            self.weights = torch.cat([self.W['y00_modes'].conj(), self.W["000_modes"], self.W["y00_modes"]],
                                     dim=-3)
            self.weights = torch.cat([self.W['yposx0_modes'].conj().rot90(k=2, dims=[-3, -2]), self.weights,
                                      self.W['yposx0_modes']], dim=-2)
            self.weights = torch.cat([self.W['yxpost_modes'].conj().rot90(k=2, dims=[-3, -2]), self.weights,
                                      self.W['yxpost_modes']], dim=-1)
        else:
            self.weights = self.W[:]

        if self.first_layer or self.last_layer:

            # construct the weight
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1)

            # apply each of the group elements to the corresponding repetition
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k].rot90(k=k, dims=[-3, -2])

            # apply each the reflection group element to the rotated kernels
            if self.reflection:
                self.weights[:, self.rt_group_size:] = self.weights[:, :self.rt_group_size].flip(dims=[-3])

            # collapse out_channels and group1 dimensions for use with conv2d
            if self.first_layer:
                self.weights = self.weights.view(-1, self.in_channels, self.kernel_size_Y, self.kernel_size_Y,
                                                 self.kernel_size_T)
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
            else:
                self.weights = self.weights.transpose(2, 1).reshape(self.out_channels, -1, self.kernel_size_Y,
                                                                    self.kernel_size_Y, self.kernel_size_T)
                self.bias = self.B

        else:

            # construct the weight
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1, 1)

            # apply elements in the rotation group
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k - 1].rot90(dims=[-3, -2])

                if self.reflection:
                    self.weights[:, k] = torch.cat([self.weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                                                    self.weights[:, k, :, :(self.rt_group_size - 1)],
                                                    self.weights[:, k, :, (self.rt_group_size + 1):],
                                                    self.weights[:, k, :, self.rt_group_size].unsqueeze(2)], dim=2)
                else:
                    self.weights[:, k] = torch.cat([self.weights[:, k, :, -1].unsqueeze(2), self.weights[:, k, :, :-1]],
                                                   dim=2)

            if self.reflection:
                # apply elements in the reflection group
                self.weights[:, self.rt_group_size:] = torch.cat(
                    [self.weights[:, :self.rt_group_size, :, self.rt_group_size:],
                     self.weights[:, :self.rt_group_size, :, :self.rt_group_size]], dim=3).flip([-3])

            # collapse out_channels / groups1 and in_channels/groups2 dimensions for use with conv3d
            self.weights = self.weights.view(self.out_channels * self.group_size, self.in_channels * self.group_size,
                                             self.kernel_size_Y, self.kernel_size_Y, self.kernel_size_T_full)
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)

        if self.Hermitian:
            self.weights = self.weights[..., -self.kernel_size_T:]

    def forward(self, x):

        self.get_weight()

        # output is of shape (batch * out_channels, number of group elements, ny, nx)
        x = nn.functional.conv3d(input=x, weight=self.weights)

        # add the bias
        if self.B is not None:
            x = x + self.bias
        return x

class GSpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, time_modes, reflection):
        super(GSpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.time_modes = time_modes
        self.conv = GConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * modes - 1,
                            kernel_size_T=2 * time_modes - 1, reflection=reflection, bias=False, spectral=True,
                            Hermitian=True)
        self.get_weight()

    # Building the weight
    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        freq0_x = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-3])) == 0).nonzero().item()
        self.get_weight()

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfftn(x, dim=[-3, -2, -1]), dim=[-3, -2])
        x_ft = x_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes),
               (freq0_x - self.modes + 1):(freq0_x + self.modes),
               :self.time_modes]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.weights.shape[0], x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes),
        (freq0_x - self.modes + 1):(freq0_x + self.modes), :self.time_modes] = self.compl_mul3d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfftn(torch.fft.ifftshift(out_ft, dim=[-3, -2]), s=(x.size(-3), x.size(-2), x.size(-1)))

        return x

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


# class GMLP3d(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels, modes, time_modes, reflection=False, last_layer=False):
#         super(GMLP3d, self).__init__()
#         self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
#         self.modes = modes
#         self.time_modes = time_modes
#         self.conv = GSpectralConv3d(in_channels=in_channels, out_channels=in_channels, modes=self.modes,
#                                      time_modes=self.time_modes, reflection=reflection)
#         self.mlp2 = GConv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, kernel_size_T=1, reflection=reflection)
#         self.w = GConv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, kernel_size_T=1, reflection=reflection)
#         # self.mlp3 = nn.Conv3d(mid_channels, out_channels, 1)
#         self.mlp3 = GConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, kernel_size_T=1,
#                             reflection=reflection, last_layer=last_layer)
#
#     def forward(self, x):
#         x = self.mlp1(x)
#         x = F.gelu(x)
#         x1 = self.conv(x)
#         x1 = self.mlp2(x1)
#         x2 = self.w(x)
#         x = x1 + x2
#         x = self.mlp3(x)
#         return x

class GMLP3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, modes, time_modes, reflection=False, last_layer=False):
        super(GMLP3d, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.modes = modes
        self.time_modes = time_modes
        self.conv = GSpectralConv3d(in_channels=in_channels, out_channels=in_channels, modes=self.modes, time_modes=self.time_modes, reflection=reflection)
        self.mlp2 = GConv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, kernel_size_T=1, reflection=reflection)
        self.w = GConv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, kernel_size_T=1, reflection=reflection)
        self.mlp3 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x1 = self.conv(x)
        x1 = self.mlp2(x1)
        x2 = self.w(x)
        x = x1 + x2
        x = self.mlp3(x)
        return x

class MLP3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP3d, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

# class GMLP3d(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels, modes, time_modes, reflection=False, last_layer=False):
#         super(GMLP3d, self).__init__()
#         self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
#         self.modes = modes
#         self.time_modes = time_modes
#         self.conv = GSpectralConv3d(in_channels=in_channels, out_channels=in_channels, modes=self.modes,
#                                      time_modes=self.time_modes, reflection=reflection)
#         self.mlp2 = GConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, kernel_size_T=1, reflection=reflection, last_layer=last_layer)
#         self.w = GConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, kernel_size_T=1, reflection=reflection, last_layer=last_layer)
#         # self.mlp3 = nn.Conv3d(mid_channels, out_channels, 1)
#
#     def forward(self, x):
#         x = self.mlp1(x)
#         x = F.gelu(x)
#         x1 = self.conv(x)
#         x = self.mlp2(x1)
#         x2 = self.w(x)
#         x = x1 + x2
#         # x = self.mlp3(x)
#         return x

class MLP3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP3d, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

# class HelmholtzProjection3D(nn.Module):
#     def __init__(
#             self,
#             n_grid: int = 64,
#             diam: float = 2 * torch.pi,
#             dtype: torch.dtype = torch.float32,
#     ):
#         super().__init__()
#         """
#         Perform Helmholtz decomposition in the frequency domain
#         to project any vector field to divergence-free.
#
#         Modified for 4D input where uhat has shape (bsz, 2, nx, ny)
#         """
#         self.n_grid = n_grid
#         self.diam = diam
#         # self._update_fft_mesh(n_grid, n_grid, diam, dtype)
#
#     def _update_fft_mesh(self, nx, ny, nt, diam=None, dtype=torch.float32):
#         diam = diam if diam is not None else self.diam
#         kx, ky, kz = fft_mesh_3d(nx, ny, nt, diam)
#         lap = spectral_laplacian_3d(fft_mesh=(kx, ky, kz))
#         self.register_buffer("lap", lap.to(dtype))
#         self.register_buffer("kx", kx.to(dtype))
#         self.register_buffer("ky", ky.to(dtype))
#         self.register_buffer("kz", kz.to(dtype))
#
#     @staticmethod
#     def div(uhat, fft_mesh):
#         """
#         uhat: (b, 2, nx, ny)
#         Returns: divergence of uhat
#         """
#         kx, ky, kz = fft_expand_dims3d(fft_mesh, uhat.size(0))
#         # print('kz', kz.shape)
#         # print('uhat[:, 2]', uhat[:, 2].shape)
#         return spectral_div_3d([uhat[:, 0], uhat[:, 1], uhat[:, 2]], (kx, ky, kz))
#
#     @staticmethod
#     def grad(uhat, fft_mesh):
#         """
#         uhat: (b, nx, ny)
#         Returns: gradient of uhat as (b, 2, nx, ny)
#         """
#         kx, ky, kz = fft_expand_dims3d(fft_mesh, uhat.size(0))
#         graduhat = spectral_grad_3d(uhat, (kx, ky, kz))
#         return torch.stack(graduhat, dim=1)
#
#     def forward(self, uhat):
#         """
#         uhat: (b, 2, nx, ny) - 4D input where the 2nd dimension is velocity components
#         """
#         bsz, _, nx, ny, nt = uhat.shape
#         # print('uhat.shape', uhat.shape)
#
#         # Update FFT mesh if the grid size changes (evaluation mode)
#         # if nx != self.n_grid:
#         self._update_fft_mesh(nx, ny, nt)
#         fft_mesh = (self.kx, self.ky, self.kz)
#
#         # Calculate divergence
#         div_u = self.div(uhat, fft_mesh)
#
#         # Calculate the gradient of the divergence
#         grad_div_u = self.grad(div_u, fft_mesh)
#
#         # Apply the Laplacian operator
#         lap = repeat(self.lap, "x y t -> b 3 x y t", b=bsz, t=nt)  # Repeat lap for the batch size
#         # lap = lap[..., :uhat.shape[-1]]
#
#         # Project the input field to make it divergence-free
#         # print('uhat',uhat.shape)
#         # print('grad_div_u', grad_div_u.shape)
#         # print('lap', lap.shape)
#         w_hat = uhat - grad_div_u / lap
#         # print('w_hat', torch.mean(w_hat))
#         # print(torch.mean(torch.abs(self.div(w_hat, fft_mesh))))
#         return w_hat
class HelmholtzProjection3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('kx', torch.zeros([192, 95, 14]))
        self.register_buffer('ky', torch.zeros([192, 95, 14]))
        self.register_buffer('kz', torch.zeros([192, 95, 14]))
        self.register_buffer('k_squared', torch.zeros([192, 95, 14]))

    def _update_fft_mesh(self, nx, ny, nt):
        gridx = torch.arange(0, 2.0 * torch.pi, 2.0 * torch.pi / nx)
        gridy = torch.linspace(1.875, 178.125, ny) / 180 * torch.pi
        radius_earth = 6.371e6
        delta_t = 15 * 60 / radius_earth
        gridz = torch.linspace(0, (nt - 1) * delta_t, nt)

        domain_size_x = gridx.max() - gridx.min()
        domain_size_y = gridy.max() - gridy.min()
        domain_size_z = gridz.max() - gridz.min()

        # nx, ny, nz = gridx.shape[1], gridx.shape[2], gridx.shape[3]

        kx = torch.fft.fftfreq(nx, d=domain_size_x/nx) * 2 * torch.pi
        ky = torch.fft.fftfreq(ny, d=domain_size_y/ny) * 2 * torch.pi
        kz = torch.fft.fftfreq(nt, d=domain_size_z/nt) * 2 * torch.pi

        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')

        kx = kx.contiguous().clone()
        ky = ky.contiguous().clone()
        kz = kz.contiguous().clone()

        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        self.register_buffer('kz', kz)
        self.register_buffer('k_squared', kx**2 + ky**2 + kz**2)

    def forward(self, u_hat):
        # print('uhat.shape', u_hat.shape)
        bsz, _, nx, ny, nt = u_hat.shape
        #         # print('uhat.shape', uhat.shape)
        #
        #         # Update FFT mesh if the grid size changes (evaluation mode)
        #         # if nx != self.n_grid:
        self._update_fft_mesh(nx, ny, nt)
        kx = self.kx.unsqueeze(0).cuda()
        ky = self.ky.unsqueeze(0).cuda()
        kz = self.kz.unsqueeze(0).cuda()
        k_squared = self.k_squared.unsqueeze(0).cuda()
        # print('u_hat',u_hat.shape)
        # print('kx', kx.shape)
        k_dot_u = u_hat[:,0,...]*kx + u_hat[:,1,...]*ky + u_hat[:,2,...]*kz
        k_squared_safe = k_squared.clone()
        k_squared_safe[k_squared_safe==0] = 1.0
        w_hat = torch.zeros_like(u_hat).cuda()
        w_hat[:,0,...] = u_hat[:,0,...] - kx * k_dot_u / k_squared_safe
        w_hat[:,1,...] = u_hat[:,1,...] - ky * k_dot_u / k_squared_safe
        w_hat[:,2,...] = u_hat[:,2,...] - kz * k_dot_u / k_squared_safe
        # print('w_hat',w_hat.shape)
        return w_hat

class SpectralConv3d_O(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_O, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.postprocess = HelmholtzProjection3D()

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # print('out_ft',out_ft.shape)

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1),
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        # print('out_ft.shape', out_ft.shape)

        out_ft = self.postprocess(out_ft)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))

        # kx, ky, kz = self.postprocess.kx.unsqueeze(0), self.postprocess.ky.unsqueeze(0), self.postprocess.kz.unsqueeze(0)
        # u_hat = torch.fft.fftn(x, dim=(-3, -2, -1))
        # print('u_hat', u_hat.shape)
        # div_hat = u_hat[:, 0, ...] * kx.cuda() + u_hat[:, 1, ...] * ky.cuda() + u_hat[:, 2, ...] * kz.cuda()
        # div = torch.fft.ifftn(div_hat, dim=(-3, -2, -1)).real
        # print('div', div.abs().mean())

        return x

class PCNO3d(nn.Module):
    def __init__(self, num_channels, modes1, modes2, modes3, width, initial_step, grid_type, time, time_pad=False):
        super(PCNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.time = time
        self.time_pad = time_pad
        self.padding = 6  # pad the domain if input is non-periodic

        self.grid = grid(twoD=False, grid_type=grid_type)
        self.p = nn.Linear(initial_step * num_channels + 3, self.width)  # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP3d(self.width, self.width, self.width)
        self.mlp1 = MLP3d(self.width, self.width, self.width)
        self.mlp2 = MLP3d(self.width, self.width, self.width)
        self.mlp3 = MLP3d(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        # self.q = MLP3d(self.width, num_channels, self.width * 4)
        self.q = GMLP3d(in_channels=self.width, out_channels=3, mid_channels=self.width * 4, modes=self.modes1, time_modes=self.modes3,
                        reflection=False, last_layer=True)  # output channel is 1: u(x, y)
        self.mc = SpectralConv3d_O(3, 3, self.modes1, self.modes2, self.modes3)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)
        grid = self.get_grid(x.shape).to(x.device)
        x = torch.cat((x, grid), dim=-1)
        # x = self.grid(x)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)

        if self.time and self.time_pad:
            x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        if self.time and self.time_pad:
            x = x[..., :-self.padding]
        x = self.q(x)
        x = self.mc(x)
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        if not self.time:  # add a time axis
            x = x.unsqueeze(-2)
        return x

    def get_grid(self, shape):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]

        gridx = torch.arange(0, 2.0 * torch.pi, 2.0 * torch.pi / size_x)
        gridy = torch.linspace(1.875, 178.125, size_y) / 180 * torch.pi
        radius_earth = 6.371e6
        delta_t = 15 * 60 / radius_earth
        gridz = torch.linspace(0, (size_z - 1) * delta_t, size_z)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])

        return torch.cat((gridx, gridy, gridz), dim=-1)