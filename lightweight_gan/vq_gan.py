from torch import nn
from lightweight_gan.modules import *
from lightweight_gan.quantizer import GumbelQuantize
from lightweight_gan.lightweight_gan import Discriminator, \
    set_requires_grad, Adam, AdaBelief, AugWrapper

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
        attn_res_layers = [],
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
            attn_res_layers = attn_res_layers
        )
        self.quantizer = GumbelQuantize(latent_dim, vocab_size, latent_dim )
        self.decoder = Decoder(
            image_size,
            input_size=downsample_size,
            latent_dim=latent_dim,
            fmap_max = fmap_max,
            fmap_inverse_coef= fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            freq_chan_attn = freq_chan_attn
        )

    def forward(self, x):
        latents = self.encoder(x)
        z_q, diff, ind = self.quantizer(latents)
        recon_x = self.decoder(z_q)
        return recon_x, z_q, diff, ind


class LightweightVQGAN(nn.Module):
    def __init__(
        self,
        latent_dim,
        image_size,
        downsample_size=32,
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 2.,
        lr = 2e-4,
        rank = 0
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            downsample_size = downsample_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
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
    input_size =8
    image_size = 128
    gen = VAE(512, image_size=image_size, downsample_size=input_size)
    latent = torch.randn(2, 128, input_size, input_size)
    print(gen(latent).shape)
