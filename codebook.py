import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from psk import PSK
from codebook_output import CodebookOutput

from typing import Sequence

class Codebook(nn.Module):

    def __init__(self, latent_dim, n_embeddings=50, beta=1e-3, codebook_train_data=None, PSNR=10):

        super().__init__()

        self.latent_dim = latent_dim
        self.n_embeddings = n_embeddings
        self.mod = PSK(M=n_embeddings, PSNR=PSNR)
        self.beta = beta
        self.train_data = codebook_train_data
        # self.layer_name = layer_name

        self.embedding = nn.Parameter(torch.Tensor(n_embeddings, latent_dim))
        nn.init.uniform_(self.embedding, -1/ n_embeddings, 1 / n_embeddings)

        # self.transmit_event = torch.cuda.Event(enable_timing=True)
        # self.receive_event = torch.cuda.Event(enable_timing=True)
        
        # self.input_index = 0

        # from torch.utils.data import TensorDataset
        # from torch.utils.data import DataLoader
        # latent = []
        # for i in range(10000):
        #     latent.append(torch.load(f".temp/Rcv_nlos_PhyDistInf_resnet56/{i}.pth"))
        # latent = torch.stack(latent, dim=0)
        # latent_ds = TensorDataset(latent)
        # self.latent_dl = DataLoader(latent_ds, batch_size=1024, num_workers=16)
        # self.batch_iter = iter(self.latent_dl)
    
    def compute_score(self, x):
        return x @ self.embedding.T / math.sqrt(self.latent_dim)
    
    def send_over_channel(self, x):
        modulated = self.mod.modulate(x)
        # raise Exception()
        # self.transmit_event.record()
        # torch.save(modulated, ".temp/latent.pth")
        received = self.mod.awgn(modulated)
        # received = next(self.batch_iter)[0].view(-1, 2).to('cuda')
        # received = torch.load(f".temp/Rcv/{self.input_index}.pth", map_location=torch.device("cuda"))
        # self.input_index += 1
        # self.receive_event.record()
        demodulated = self.mod.demodulate(received)
        return demodulated
    
    def construct_noise(self, samples):
        x = samples.argmax(dim=-1)
        x_tilde = self.send_over_channel(x)
        noise = F.one_hot(x_tilde, num_classes=self.n_embeddings).float() -\
            F.one_hot(x, num_classes=self.n_embeddings).float()
        return noise
    
    def sample(self, score):
        dist = score.softmax(dim=-1)
        if self.training:
            samples = F.gumbel_softmax(score, tau=0.5, hard=True)
            noise = self.construct_noise(samples)
            samples = samples + noise
        else:
            samples = score.argmax(dim=-1)
            samples = self.send_over_channel(samples)
            samples = F.one_hot(samples, num_classes=self.n_embeddings).float()

        return samples, dist

    def forward(self, x):
        original_codebook_outputs = x
        was_codebook = type(x) == CodebookOutput

        if was_codebook:
            x = x.original_tensor

        initial_shape = x.shape
        if len(initial_shape) > 2:
            x_reshaped = x.view(-1, self.latent_dim)
        score = self.compute_score(x_reshaped)
        samples, dist = self.sample(score)

        res = samples @ self.embedding

        if len(initial_shape) > 2:
            res = res.view(*initial_shape)

        if was_codebook:
            codebook_outputs = original_codebook_outputs.codebook_outputs
            codebook_outputs.append([res, dist, self])
            output = CodebookOutput(original_codebook_outputs.original_tensor, codebook_outputs)
            return output
        else:
            output = CodebookOutput(x, [[res, dist, self]])
            return output

    def __repr__(self):
        return f'Codebook(latent_dim={self.latent_dim}, n_embeddings={self.n_embeddings})'
    