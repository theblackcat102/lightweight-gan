from torch import nn
from lightweight_gan.modules import *
from lightweight_gan.quantizer import GumbelQuantize
from lightweight_gan.lightweight_gan import Discriminator, \
    set_requires_grad, Adam, AdaBelief, AugWrapper, get_dct_weights
from lightweight_gan.lpips import LPIPS

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight




class VAE(nn.Module):
    def __init__(self, 
        latent_dim,
        image_size,
        vocab_size=16384,
        downsample_size=32,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        enc_attn_res_layers = [],
        dec_attn_res_layers = [],
        freq_chan_attn = False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.encoder = Encoder(
            image_size=self.image_size,
            downsample=downsample_size,
            n_hid=latent_dim,
            fmap_max = fmap_max,
            fmap_inverse_coef= fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = enc_attn_res_layers
        )
        freq_w, freq_h = ([0] * 8), list(range(8)) # in paper, it seems 16 frequencies was ideal
        dct_weights = get_dct_weights(downsample_size, latent_dim, [*freq_w, *freq_h], [*freq_h, *freq_w])
        self.register_buffer('dct_weights', dct_weights)

        self.quantizer = GumbelQuantize(latent_dim, vocab_size, latent_dim )
        self.decoder = Decoder(
            image_size,
            input_size=downsample_size,
            latent_dim=latent_dim,
            fmap_max = fmap_max,
            fmap_inverse_coef= fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = dec_attn_res_layers,
            freq_chan_attn = freq_chan_attn
        )

    def forward(self, x, temp=None):
        latents = self.encoder(x)
        z_q, diff, ind = self.quantizer(latents, temp=temp)
        recon_x = self.decoder(z_q * self.dct_weights)
        return recon_x, z_q, diff, ind


class LightweightVQGAN(nn.Module):
    def __init__(
        self,
        latent_dim,
        image_size,
        downsample_size=32,
        optimizer = "adam",
        vocab_size=16384,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        dec_attn_res_layers=[],
        enc_attn_res_layers=[],
        freq_chan_attn = False,
        ttur_mult = 2.,
        perceptual_weight=1.0, # adaptive weight
        disc_weight=1.0,# adaptive weight
        discriminator_iter_start=25001,
        lr = 2e-4,
        rank = 0
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            vocab_size=vocab_size,
            downsample_size = downsample_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            enc_attn_res_layers = enc_attn_res_layers,
            dec_attn_res_layers = dec_attn_res_layers,
            freq_chan_attn = freq_chan_attn
        )

        self.G = VAE(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )
        if perceptual_weight > 0:
            self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight
        self.discriminator_iter_start = discriminator_iter_start

        self.ema_updater = EMA(0.995)
        self.GE = VAE(**G_kwargs)
        set_requires_grad(self.GE, False)

        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        elif optimizer == "adabelief":
            self.G_opt = AdaBelief(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = AdaBelief(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)
    
    def get_last_layer(self):
        return self.G.decoder.out_conv.weight if isinstance(self.G, nn.DataParallel) else self.G.module.decoder.out_conv.weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.get_last_layer()[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.get_last_layer()[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented


if __name__ == '__main__':
    input_size = 32
    image_size = 512
    gen = LightweightVQGAN(512, image_size=image_size, 
        downsample_size=input_size, 
        dec_attn_res_layers=[16, 32], 
        freq_chan_attn=True, 
        perceptual_weight=-1)
    print(gen.state_dict().keys())
    print('VAE',sum(p.numel() for p in gen.parameters() )/1e6)
    D = gen.D
    print('D',sum(p.numel() for p in D.parameters() )/1e6)

    img = torch.randn(2, 3, image_size, image_size)
    fake_img = gen.G(img)[0]
    print(fake_img.shape)

    out, out_32x32, aux_loss = D(fake_img)
    print(out.shape)
