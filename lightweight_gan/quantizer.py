"""
The critical quantization layers that we sandwich in the middle of the autoencoder
(between the encoder and decoder) that force the representation through a categorical
variable bottleneck and use various tricks (softening / straight-through estimators)
to backpropagate through the sampling process.
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F

from scipy.cluster.vq import kmeans2

# -----------------------------------------------------------------------------

class VQVAEQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937
    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.kld_scale = 10.0

        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.register_buffer('data_initialized', torch.zeros(1))

    def forward(self, z_e):
        B, _, H, W = z_e.size()

        # project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
        z_e = z_e.permute(0, 2, 3, 1) # make (B, H, W, C)
        flatten = z_e.reshape(-1, self.embedding_dim)

        # DeepMind def does not do this but I find I have to... ;\
        if self.training and self.data_initialized.item() == 0:
            print('running kmeans!!') # data driven initialization for the embeddings
            rp = torch.randperm(flatten.size(0))
            kd = kmeans2(flatten[rp[:20000]].data.cpu().numpy(), self.n_embed, minit='points')
            self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)
            # TODO: this won't work in multi-GPU setups

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        _, ind = (-dist).max(1)
        ind = ind.view(B, H, W)

        # vector quantization cost that trains the embedding vectors
        z_q = self.embed_code(ind) # (B, H, W, C)
        commitment_cost = 0.25
        diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
        diff *= self.kld_scale

        z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass
        z_q = z_q.permute(0, 3, 1, 2) # stack encodings into channels again: (B, C, H, W)
        return z_q, diff, ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


class GumbelQuantize(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, input_dim, n_embed, embedding_dim, straight_through=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-4
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, input_dim//4, 1),
            nn.ReLU(),
            nn.Conv2d(input_dim//4, n_embed, 1)
        )
        self.embed = nn.Embedding(n_embed, embedding_dim)

    def forward(self, z):
        logits = self.proj(z)
        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if self.training else True

        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        return z_q, diff, ind


import math
import torch

# -----------------------------------------------------------------------------

class LogitLaplace:
    """ the Logit Laplace distribution log likelihood from OpenAI's DALL-E paper """
    logit_laplace_eps = 0.1

    @classmethod
    def inmap(cls, x):
        # map [0,1] range to [eps, 1-eps]
        return (1 - 2 * cls.logit_laplace_eps) * x + cls.logit_laplace_eps

    @classmethod
    def unmap(cls, x):
        # inverse map, from [eps, 1-eps] to [0,1], with clamping
        return torch.clamp((x - cls.logit_laplace_eps) / (1 - 2 * cls.logit_laplace_eps), 0, 1)

    @classmethod
    def nll(cls, x, mu_logb):
        raise NotImplementedError # coming right up


class Normal:
    """
    simple normal distribution with fixed variance, as used by DeepMind in their VQVAE
    note that DeepMind's reconstruction loss (I think incorrectly?) misses a factor of 2,
    which I have added to the normalizer of the reconstruction loss in nll(), we'll report
    number that is half of what we expect in their jupyter notebook
    """
    data_variance = 0.06327039811675479 # cifar-10 data variance, from deepmind sonnet code

    @classmethod
    def inmap(cls, x):
        return x - 0.5 # map [0,1] range to [-0.5, 0.5]

    @classmethod
    def unmap(cls, x):
        return torch.clamp(x + 0.5, 0, 1)

    @classmethod
    def nll(cls, x, mu):
        return ((x - mu)**2).mean() / (2 * cls.data_variance) #+ math.log(math.sqrt(2 * math.pi * cls.data_variance))

if __name__ == '__main__':
    from lightweight_gan.modules import Encoder, Decoder
    input_size = 32
    image_size = 512
    # gen = Decoder(image_size, latent_dim=128, input_size=input_size )
    # latent = torch.randn(2, 128, input_size, input_size)
    # print(gen(latent).shape)

    input_size = 32
    image_size = 512

    enc = Encoder(image_size, downsample=32, attn_res_layers=[64, 32])
    x = torch.randn((2, 3, image_size, image_size))

    latents = enc(x)
    print(latents.shape)
    gen = Decoder(image_size, latent_dim=768, input_size=input_size, attn_res_layers=[64] )
    latent = torch.randn(2, 128, input_size, input_size)

    quantize = GumbelQuantize(768, 16384, 768 )
    z_q, diff, ind = quantize(latents)
    print(z_q.shape, diff, ind.shape)
    print(gen(z_q).shape)
    print('enc',sum(p.numel() for p in enc.parameters() if p.requires_grad)/1e6)
    print('dec',sum(p.numel() for p in gen.parameters() if p.requires_grad)/1e6)
    print('quantize',sum(p.numel() for p in quantize.parameters() if p.requires_grad)/1e6)

