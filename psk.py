import math
import torch
import torch.nn as nn

class PSK(nn.Module):

    def __init__(self, M, PSNR):
        super().__init__()

        self.M = M
        self.PSNR = PSNR
        self.p = 1

        sins = (torch.arange(M) * 2 * torch.pi / M).sin().unsqueeze(1)
        coss = (torch.arange(M) * 2 * torch.pi / M).cos().unsqueeze(1)
        constellation = torch.cat([coss, sins], dim=1)
        self.register_buffer('constellation', constellation)

        self.noise_scale = math.sqrt(0.5 * self.p / math.pow(10, self.PSNR / 10))

    def modulate(self, z):
        return self.constellation[z]
    
    def demodulate(self, x):
        return (x @ self.constellation.T).argmax(dim=-1)
    
    def awgn(self, x):
        return x + torch.randn_like(x) * self.noise_scale