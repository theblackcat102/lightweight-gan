from PIL.Image import Image
from torch.utils import data
from tqdm import tqdm
from datetime import datetime
from functools import wraps
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
import random
import os, math
import torchvision
from torchvision import transforms

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lightweight_gan.diff_augment_test import DiffAugmentTest
from lightweight_gan.lightweight_gan import ImageDataset, \
    GradScaler, \
    dual_contrastive_loss, hinge_loss, cycle, default, gen_hinge_loss, \
    autocast, null_context, safe_div, raise_if_nan, torch_grad
from absl import flags, app
from lightweight_gan.vq_gan import LightweightVQGAN

FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'default', 'Text to echo.')
flags.DEFINE_string('data', './data', 'data directory')
flags.DEFINE_float('learning_rate', 2e-4, 'learning rate')
flags.DEFINE_float('ttur_mult', 2.0, 'TTUR multiplication for dis')

flags.DEFINE_integer('batch_size', 10, 'training batch size')
flags.DEFINE_integer('image_size', 512, 'image size')
flags.DEFINE_integer('downsample', 32, 'image down sample to')
flags.DEFINE_integer('gradient_accumulate_every', 1, 'gradient accumulate every')
flags.DEFINE_integer('num_train_steps', 100000, 'training iteration')
flags.DEFINE_integer('disc_output_size', 1, 'discriminator output size')
flags.DEFINE_boolean('dual_contrast_loss', False, 'dual constrastive loss for discriminator')

flags.DEFINE_multi_string('aug_types', ['cutout', 'translation'], 'augmentation type')
flags.DEFINE_float('aug_prob', 0, 'augmentation probability')
flags.DEFINE_float('dataset_aug_prob', 0.1, 'dataset augmentation probability')

flags.DEFINE_multi_integer('attn_res_layers', [], 'attention')
flags.DEFINE_multi_integer('enc_attn_res_layers', [], 'encoder attention')
flags.DEFINE_multi_integer('dec_attn_res_layers', [], 'decoder attention')
flags.DEFINE_multi_integer('dis_attn_res_layers', [], 'discriminator attention')

flags.DEFINE_boolean('freq_chan_attn', False, 'image freq channel attention')

# model parameters
flags.DEFINE_integer('latent_dim', 768, 'latent dimension')
# flags.DEFINE_integer('batch_size', 64, 'training batch size')
flags.DEFINE_float('gp_weight', 10, 'augmentation probability')
flags.DEFINE_float('recon_weight', 1, 'vae reconstruct probability')

flags.DEFINE_boolean('transparent', False, 'transparent?')
flags.DEFINE_boolean('smooth_l1_loss', False, 'VAE loss function l1 or mse')

flags.DEFINE_boolean('apply_gradient_penalty', True, 'transparent?')
flags.DEFINE_boolean('greyscale', False, 'grayscale?')
flags.DEFINE_boolean('amp', False, 'use amp')

flags.DEFINE_string('optimizer', 'adam', 'optimizer')

flags.DEFINE_integer('num_workers', 10, 'number of worker')


def step_dis(VQGAN, image_batch, amp_context,L_scaler, device='cuda', ):
    aug_prob   = default(FLAGS.aug_prob, 0)
    aug_types  = FLAGS.aug_types
    aug_kwargs = {'prob': aug_prob, 'types': aug_types}

    total_disc_loss = torch.zeros([], device=device)
    G = VQGAN.G
    D_aug = VQGAN.D_aug
    if FLAGS.dual_contrast_loss:
        D_loss_fn = dual_contrastive_loss
    else:
        D_loss_fn = hinge_loss

    # grad acc here
    for _ in range(FLAGS.gradient_accumulate_every):
        with amp_context():
            with torch.no_grad():
                generated_images = G(image_batch)
            fake_output, fake_output_32x32, _ = D_aug(generated_images[0], detach = True, **aug_kwargs)

            real_output, real_output_32x32, real_aux_loss = D_aug(image_batch,  calc_aux_loss = True, **aug_kwargs)

            real_output_loss = real_output
            fake_output_loss = fake_output
            divergence = D_loss_fn(real_output_loss, fake_output_loss)
            divergence_32x32 = D_loss_fn(real_output_32x32, fake_output_32x32)
            disc_loss = divergence + divergence_32x32

            aux_loss = real_aux_loss
            disc_loss = disc_loss + aux_loss

        if FLAGS.apply_gradient_penalty:
            outputs = [real_output, real_output_32x32]
            outputs = list(map(L_scaler.scale, outputs)) if FLAGS.amp else outputs

            scaled_gradients = torch_grad(outputs=outputs, inputs=image_batch,
                                    grad_outputs=list(map(lambda t: torch.ones(t.size(), device = image_batch.device), outputs)),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

            inv_scale = safe_div(1., L_scaler.get_scale()) if FLAGS.amp else 1.

            if inv_scale != float('inf'):
                gradients = scaled_gradients * inv_scale

                with amp_context():
                    gradients = gradients.reshape(FLAGS.batch_size, -1)
                    gp = FLAGS.gp_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                    if not torch.isnan(gp):
                        disc_loss = disc_loss + gp
                        last_gp_loss = gp.clone().detach().item()

        with amp_context():
            disc_loss = disc_loss / FLAGS.gradient_accumulate_every

        disc_loss.register_hook(raise_if_nan)
        L_scaler.scale(disc_loss).backward()
        total_disc_loss += divergence
    return total_disc_loss / FLAGS.gradient_accumulate_every

def step_gen(VQGAN, image_batch, amp_context, L_scaler, device='cuda'):
    aug_prob   = default(FLAGS.aug_prob, 0)
    aug_types  = FLAGS.aug_types
    aug_kwargs = {'prob': aug_prob, 'types': aug_types}
    total_gen_loss = torch.zeros([], device=device)
    
    G = VQGAN.G
    D_aug = VQGAN.D_aug
    if FLAGS.dual_contrast_loss:
        G_loss_fn = dual_contrastive_loss
        G_requires_calc_real = True
    else:
        G_loss_fn = gen_hinge_loss
        G_requires_calc_real = False
    recon_loss_fn = F.smooth_l1_loss if FLAGS.smooth_l1_loss else F.mse_loss
    for _ in range(FLAGS.gradient_accumulate_every):

        with amp_context():
            generated_images, _, latent_loss, _ = G(image_batch)

            fake_output, fake_output_32x32, _ = D_aug(generated_images, **aug_kwargs)
            real_output, real_output_32x32, _ = D_aug(image_batch, **aug_kwargs) if G_requires_calc_real else (None, None, None)

            loss = G_loss_fn(fake_output, real_output)
            loss_32x32 = G_loss_fn(fake_output_32x32, real_output_32x32)

            gen_loss = loss + loss_32x32 + recon_loss_fn(image_batch, generated_images) * FLAGS.recon_weight \
                + latent_loss

            gen_loss = gen_loss / FLAGS.gradient_accumulate_every

        gen_loss.register_hook(raise_if_nan)
        L_scaler.scale(gen_loss).backward()
        total_gen_loss += loss

    return total_gen_loss / FLAGS.gradient_accumulate_every

def train(argv):
    dataset = ImageDataset('/mnt/ws/ffhq-dataset/images1024x1024/', FLAGS.image_size, aug_prob=FLAGS.dataset_aug_prob)
    dataloader = DataLoader(dataset, 
        num_workers = FLAGS.num_workers, 
        batch_size = FLAGS.batch_size, 
        shuffle = True, drop_last = True, pin_memory = True)

    loader = cycle(dataloader)

    VQGAN = LightweightVQGAN(
        latent_dim=FLAGS.latent_dim,
        image_size=FLAGS.image_size,
        downsample_size=FLAGS.downsample,
        attn_res_layers=FLAGS.attn_res_layers,
        freq_chan_attn=FLAGS.freq_chan_attn,
        ttur_mult = FLAGS.ttur_mult,
        lr=FLAGS.learning_rate
    )

    amp = FLAGS.amp
    amp_context = autocast if amp else null_context
    G_scaler = GradScaler(enabled = amp)
    D_scaler = GradScaler(enabled = amp)

    VQGAN.D_opt.zero_grad()
    for i in range(100000):

        image_batch = next(loader).cuda()
        image_batch.requires_grad_()
        VQGAN.D_opt.zero_grad()
        dis_loss = step_dis(VQGAN, image_batch, amp_context, D_scaler )
        D_scaler.step(VQGAN.D_opt)
        D_scaler.update()

        image_batch = next(loader).cuda()
        image_batch.requires_grad_()

        VQGAN.G_opt.zero_grad()
        gen_loss = step_gen(VQGAN, image_batch, amp_context, G_scaler)
        print(i, gen_loss.item(), dis_loss.item())
        G_scaler.step(VQGAN.G_opt)
        G_scaler.update()

        if i % 10 == 0 and i > 20000:
            VQGAN.EMA()

        if i % 500 == 0:
            generated_images = []
            total = 0
            num_rows = 8
            with torch.no_grad():
                while total < (num_rows*num_rows):
                    image_batch = next(loader).cuda()
                    total += len(image_batch[0])
                    generated_images.append(VQGAN.G(image_batch)[0].cpu())
            generated_images = torch.cat(generated_images)
            images_grid = torchvision.utils.make_grid(generated_images[:int(num_rows**2)], nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            pil_image.save('results/vqgan/{}.jpg'.format(i//500))


if __name__ == '__main__':
  app.run(train)