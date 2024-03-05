import numpy as np

import torch
import torch.nn as nn


@torch.jit.script
def compl_mul3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixyz,ioxyz->boxyz", a, b)
    return res


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3,4])
        
        z_dim = min(x_ft.shape[4], self.modes3)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[2], x_ft.shape[3], self.modes3, device=x.device, dtype=torch.cfloat)
        
        # if x_ft.shape[4] > self.modes3, truncate; if x_ft.shape[4] < self.modes3, add zero padding 
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, :self.modes1, :self.modes2, :z_dim]
        out_ft[:, :, :self.modes1, :self.modes2, :] = compl_mul3d(coeff, self.weights1)
        
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, :self.modes2, :z_dim]
        out_ft[:, :, -self.modes1:, :self.modes2, :] = compl_mul3d(coeff, self.weights2)
        
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, :self.modes1, -self.modes2:, :z_dim]
        out_ft[:, :, :self.modes1, -self.modes2:, :] = compl_mul3d(coeff, self.weights3)
        
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, -self.modes2:, :z_dim]
        out_ft[:, :, -self.modes1:, -self.modes2:, :] = compl_mul3d(coeff, self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4)), dim=[2,3,4])
        return x

