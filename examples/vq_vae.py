import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True
from torch import nn

from PIL.Image import Image
from torch.utils import data
from tqdm import tqdm
from datetime import datetime
from functools import wraps
import numpy as np
import random
import os, math
import torchvision
from tqdm import trange
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from lightweight_gan.diff_augment_test import DiffAugmentTest
from lightweight_gan.lightweight_gan import ImageDataset, \
    GradScaler, \
    dual_contrastive_loss, hinge_loss, cycle, default, gen_hinge_loss, \
    autocast, null_context, safe_div, raise_if_nan, torch_grad
from absl import flags, app
from lightweight_gan.vq_gan import LightweightVQGAN, adopt_weight
from tensorboardX import SummaryWriter


FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'default', 'Text to echo.')
flags.DEFINE_list('data', './data', 'data directory')
flags.DEFINE_float('learning_rate', 2e-4, 'learning rate')
flags.DEFINE_float('ttur_mult', 1.0, 'TTUR multiplication for dis')

flags.DEFINE_integer('batch_size', 10, 'training batch size')
flags.DEFINE_integer('image_size', 512, 'image size')
flags.DEFINE_integer('downsample', 32, 'image down sample to')
flags.DEFINE_integer('gradient_accumulate_every', 1, 'gradient accumulate every')
flags.DEFINE_integer('dis_step', 1, 'gradient accumulate every')

flags.DEFINE_integer('num_train_steps', 100000, 'training iteration')
flags.DEFINE_integer('warmup_steps', 1000, 'training iteration')

flags.DEFINE_integer('fmap_max', 512, 'vae channel size')
flags.DEFINE_integer('d_fmap_max', 512, 'discriminator channel size')

flags.DEFINE_integer('disc_output_size', 1, 'discriminator output size')
flags.DEFINE_boolean('dual_contrast_loss', False, 'dual constrastive loss for discriminator')

flags.DEFINE_multi_string('aug_types', ['cutout', 'translation'], 'augmentation type')
flags.DEFINE_float('aug_prob', 0, 'augmentation probability')
flags.DEFINE_float('dataset_aug_prob', 0.1, 'dataset augmentation probability')

flags.DEFINE_multi_integer('attn_res_layers', [], 'attention')
flags.DEFINE_multi_integer('enc_attn_res_layers', [], 'encoder attention')
flags.DEFINE_multi_integer('dec_attn_res_layers', [], 'decoder attention')
flags.DEFINE_multi_integer('dis_attn_res_layers', [], 'discriminator attention')
flags.DEFINE_boolean('recon_only', False, 'image freq channel attention')
flags.DEFINE_boolean('freq_chan_attn', False, 'image freq channel attention')
flags.DEFINE_integer('discriminator_iter_start', 25000, 'when to start discriminator loss')

# model parameters
flags.DEFINE_integer('latent_dim', 768, 'latent dimension')
flags.DEFINE_integer('vocab_size', 16384, 'latent dimension')
# flags.DEFINE_integer('batch_size', 64, 'training batch size')
flags.DEFINE_float('gp_weight', 10, 'augmentation probability')
flags.DEFINE_float('recon_weight', 1, 'vae reconstruct probability')

flags.DEFINE_integer('sample_grid_size', 8, 'grid size sample')

flags.DEFINE_float('init_temp', 1.0, 'initial anneal temperature')
flags.DEFINE_float('anneal_rate', 0.00005, 'initial anneal temperature')
flags.DEFINE_float('min_temp', 0.5, 'minimum anneal temperature')
flags.DEFINE_integer('update_temp', 100, 'when to update temperature')

flags.DEFINE_float('perceptual_weight', 1.0, 'minimum anneal temperature')
flags.DEFINE_float('c', 0.01, 'minimum anneal temperature')
flags.DEFINE_float('disc_weight', 1.0, 'adaptive dis weight factor')
flags.DEFINE_string('dis_type', 'lightweight', 'adaptive dis weight factor')

flags.DEFINE_boolean('detach_z', False, 'detach vae latent')

flags.DEFINE_boolean('transparent', False, 'transparent?')
flags.DEFINE_boolean('smooth_l1_loss', True, 'VAE loss function l1 or mse')

flags.DEFINE_boolean('apply_gradient_penalty', True, 'transparent?')
flags.DEFINE_boolean('greyscale', False, 'grayscale?')
flags.DEFINE_boolean('amp', False, 'use amp')
flags.DEFINE_boolean('viz', False, 'use amp')

flags.DEFINE_string('optimizer', 'adam', 'optimizer')
flags.DEFINE_integer('num_gpus', 1, 'number of GPUs')
flags.DEFINE_integer('sample_every_steps', 1000, 'number of steps to sample image')
flags.DEFINE_integer('checkpoint_every_steps', 2000, 'number of checkpoints')
flags.DEFINE_string('checkpoint', None, 'start from checkpoint')


flags.DEFINE_integer('num_workers', 10, 'number of worker')


def lr_lambda(current_step):
    learning_rate = max(0.0, 1. - (float(current_step) / float(FLAGS.num_train_steps)))
    if FLAGS.warmup_steps > 0:
        learning_rate *= min(1.0, float(current_step) / float(FLAGS.warmup_steps))
    return learning_rate



def step_dis(step, VQGAN, image_batch, amp_context,L_scaler, device='cuda', ):
    aug_prob   = default(FLAGS.aug_prob, 0)
    aug_types  = FLAGS.aug_types
    aug_kwargs = {'prob': aug_prob, 'types': aug_types}
    logs = {}

    total_disc_loss = torch.zeros([], device=device)
    G = VQGAN.G
    D_aug = VQGAN.D_aug
    if FLAGS.dual_contrast_loss:
        D_loss_fn = dual_contrastive_loss
    else:
        D_loss_fn = hinge_loss

    # grad acc here
    for _ in range(FLAGS.gradient_accumulate_every * FLAGS.dis_step):
        with amp_context():
            with torch.no_grad():
                generated_images = G(image_batch)

            if FLAGS.dis_type == 'lightweight':
                fake_output, fake_output_32x32, _ = D_aug(
                    generated_images[0], detach = True, **aug_kwargs)

                real_output, real_output_32x32, real_aux_loss = D_aug(
                    image_batch,  calc_aux_loss = True, **aug_kwargs)

                real_output_loss = real_output
                fake_output_loss = fake_output
                divergence = D_loss_fn(real_output_loss, fake_output_loss)

                logs['D/divergence'] = divergence.item()
                divergence_32x32 = D_loss_fn(real_output_32x32, fake_output_32x32)
                disc_loss = divergence + divergence_32x32

                aux_loss = real_aux_loss
                logs['D/aux'] = real_aux_loss.item()
                disc_loss = disc_loss + aux_loss
            else:
                fake_output = D_aug(generated_images[0])
                real_output =  D_aug(image_batch)

                divergence = D_loss_fn(real_output, fake_output)
                disc_loss = divergence
                logs['D/divergence'] = disc_loss.item()

        if FLAGS.apply_gradient_penalty:
            print('apply grad')
            if FLAGS.dis_type == 'lightweight':
                outputs = [real_output, real_output_32x32]
            else:
                outputs = [real_output]
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
                    logs['D/gp'] = gp.item()
                    if not torch.isnan(gp):
                        disc_loss = disc_loss + gp
                        last_gp_loss = gp.clone().detach().item()

        with amp_context():
            disc_loss = disc_loss / (FLAGS.gradient_accumulate_every * FLAGS.dis_step)

        disc_loss.register_hook(raise_if_nan)
        L_scaler.scale(disc_loss).backward()
        total_disc_loss += divergence
    return total_disc_loss / (FLAGS.gradient_accumulate_every * FLAGS.dis_step), logs

def step_gen(step, VQGAN, image_batch, amp_context, L_scaler, device='cuda', recon_only=False, temp=1.0):
    aug_prob   = default(FLAGS.aug_prob, 0)
    aug_types  = FLAGS.aug_types
    aug_kwargs = {'prob': aug_prob, 'types': aug_types}
    total_gen_loss = torch.zeros([], device=device)
    logs = {}
    
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
            loss = recon_loss_fn(generated_images, image_batch )
            if FLAGS.perceptual_weight > 0:
                p_loss = VQGAN.perceptual_loss(image_batch.contiguous(), generated_images.contiguous()).mean()
                loss = loss + FLAGS.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor([0.0])
          
            logs['G/recon'] = loss.item()
            gen_loss = loss * FLAGS.recon_weight \
                + latent_loss.mean()
            if not recon_only and step > FLAGS.discriminator_iter_start:
                # generated_images = G(z_q.detach() , decode_only=True)
                if FLAGS.dis_type == 'lightweight':
                    fake_output, fake_output_32x32, _ = D_aug(generated_images, **aug_kwargs)
                    real_output, real_output_32x32, _ = D_aug(image_batch, **aug_kwargs) if G_requires_calc_real else (None, None, None)
                    loss_32x32 = G_loss_fn(fake_output_32x32, real_output_32x32)

                else:
                    fake_output = D_aug(generated_images)
                    real_output = D_aug(image_batch)

                g_loss = G_loss_fn(fake_output, real_output)
                if FLAGS.dis_type == 'lightweight':
                    g_loss += loss_32x32

                d_weight = VQGAN.calculate_adaptive_weight(gen_loss, g_loss)
                disc_factor = adopt_weight(VQGAN.disc_factor, step, threshold=FLAGS.discriminator_iter_start)

                logs['G/d_weight'] = d_weight.item()
                logs['G/g_loss'] = g_loss.item()
                logs['G/disc_factor'] = disc_factor

                gen_loss += g_loss  * d_weight * disc_factor
            gen_loss = gen_loss / FLAGS.gradient_accumulate_every

        gen_loss.register_hook(raise_if_nan)
        L_scaler.scale(gen_loss).backward()
        total_gen_loss += loss

    return total_gen_loss / FLAGS.gradient_accumulate_every, logs

def store_sample(loader, save_filename):
    generated_images = []
    total = 0
    num_rows = 8
    with torch.no_grad():
        while total < (num_rows*num_rows):
            image_batch = next(loader)
            generated_images.append(image_batch )
            total += len(image_batch[0])

    generated_images_tensor = torch.cat(generated_images)
    images_grid = torchvision.utils.make_grid(generated_images_tensor.float().clamp_(0., 1.)[:int(num_rows**2)], nrow = num_rows)
    pil_image = transforms.ToPILImage()(images_grid.cpu())
    pil_image.save(save_filename)

    return generated_images

def train():
    if len(FLAGS.data) == 1:
        dataset = ImageDataset(FLAGS.data[0], FLAGS.image_size, aug_prob=FLAGS.dataset_aug_prob)
    else:
        print(FLAGS.data)
        dataset = ConcatDataset(
            [ 
                ImageDataset(dataset_path, FLAGS.image_size, aug_prob=FLAGS.dataset_aug_prob) \
                for dataset_path in FLAGS.data  ]
        )

    dataloader = DataLoader(dataset, 
        num_workers = FLAGS.num_workers, 
        batch_size = FLAGS.batch_size, 
        shuffle = True, drop_last = True, pin_memory = True)
    print(len(dataloader))

    loader = cycle(dataloader)

    VQGAN = LightweightVQGAN(
        latent_dim=FLAGS.latent_dim,
        image_size=FLAGS.image_size,
        downsample_size=FLAGS.downsample,
        vocab_size=FLAGS.vocab_size,
        attn_res_layers=FLAGS.attn_res_layers,
        freq_chan_attn=FLAGS.freq_chan_attn,
        perceptual_weight=FLAGS.perceptual_weight,
        enc_attn_res_layers=FLAGS.enc_attn_res_layers,
        dec_attn_res_layers=FLAGS.dec_attn_res_layers,
        ttur_mult = FLAGS.ttur_mult,
        fmap_max=FLAGS.fmap_max,
        d_fmap_max=FLAGS.d_fmap_max,
        disc_weight=FLAGS.disc_weight,
        disc_type=FLAGS.dis_type,
        optimizer=FLAGS.optimizer,
        lr=FLAGS.learning_rate,
        discriminator_iter_start=FLAGS.discriminator_iter_start,
    )

    amp = FLAGS.amp
    amp_context = autocast if amp else null_context
    G_scaler = GradScaler(enabled = amp)
    D_scaler = GradScaler(enabled = amp)
    start_step = 0

    D_scheduler = LambdaLR(VQGAN.D_opt, lr_lambda, last_epoch=-1)
    G_scheduler = LambdaLR(VQGAN.G_opt, lr_lambda, last_epoch=-1)

    if FLAGS.checkpoint is not None:
        state_dict = torch.load(FLAGS.checkpoint, map_location='cuda')
        VQGAN.G.load_state_dict(state_dict['G'])
        # disable these seems to work for GAN stage

        VQGAN.G_opt.load_state_dict(state_dict['G_opt'])
        if amp:
            G_scaler.load_state_dict(state_dict['G_scaler'])
        G_scheduler.load_state_dict(state_dict['G_scheduler'])

        VQGAN.D.load_state_dict(state_dict['D'])

        VQGAN.D_opt.load_state_dict(state_dict['D_opt'])
        # D_scaler.load_state_dict(state_dict['D_scaler'])
        D_scheduler.load_state_dict(state_dict['D_scheduler'])
        if 'GE' not in state_dict:
            print('reset EMA')
            VQGAN.reset_parameter_averaging()

        start_step = state_dict['step']
        del state_dict

    if FLAGS.num_gpus > 1:
        VQGAN.G = nn.DataParallel(VQGAN.G, device_ids=list(range(FLAGS.num_gpus)))
        VQGAN.D = nn.DataParallel(VQGAN.D, device_ids=list(range(FLAGS.num_gpus)))


    temp = FLAGS.init_temp
    img_samples = store_sample(loader, 'results/{}/{}.jpg'.format(FLAGS.name, 'samples'))
    with open(os.path.join('results/{}'.format(FLAGS.name), "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    D_opt = VQGAN.D_opt 
    G_opt = VQGAN.G_opt

    recon_only = FLAGS.recon_only

    writer = SummaryWriter('results/{}'.format(FLAGS.name))
    '''
    Tensorboard loggging
    '''

    with trange(start_step, FLAGS.num_train_steps + 1, desc='Training', ncols=0, dynamic_ncols=True) as pbar:
        for step in range(start_step, FLAGS.num_train_steps):

            if step > FLAGS.discriminator_iter_start:
                recon_only = False

            if not recon_only:
                image_batch = next(loader).cuda()
                image_batch.requires_grad_()
                D_opt.zero_grad()
                dis_loss, logs = step_dis(step, VQGAN, image_batch, amp_context, D_scaler )
                for key, item in logs.items():
                    writer.add_scalar(key, item, step)
                D_scaler.step(D_opt)
                D_scaler.update()

                D_scheduler.step()

            image_batch = next(loader).cuda()
            image_batch.requires_grad_()

            G_opt.zero_grad()

            if step % FLAGS.update_temp == 0:
                temp = np.maximum(FLAGS.init_temp * np.exp(-FLAGS.anneal_rate * step), FLAGS.min_temp)
            gen_loss, logs = step_gen(step, VQGAN, image_batch, amp_context, G_scaler, recon_only=recon_only, temp=temp)
            for key, item in logs.items():
                writer.add_scalar(key, item, step)

            if recon_only:
                pbar.set_postfix(gloss='%.4f' % gen_loss.item(), 
                    temp='%.4f' % temp)
            else:
                pbar.set_postfix(gloss='%.4f' % gen_loss.item(), 
                    dloss='%.4f' % dis_loss.item(),
                    temp='%.4f'  % temp)

            G_scaler.step(G_opt)
            G_scaler.update()
            G_scheduler.step()

            if step % 10 == 0 and step > 20000:
                VQGAN.EMA()

            if step % FLAGS.sample_every_steps == 0:
                generated_images = []
                total = 0
                num_rows = FLAGS.sample_grid_size
                with torch.no_grad():
                    for image_batch in img_samples:
                        image_batch = image_batch.cuda()
                        generated_images.append(VQGAN.G(image_batch)[0].cpu().float().clamp_(0., 1.) )

                        total += len(image_batch[0])

                generated_images = torch.cat(generated_images)
                images_grid = torchvision.utils.make_grid(generated_images[:int(num_rows**2)], nrow = num_rows)
                pil_image = transforms.ToPILImage()(images_grid.cpu())
                pil_image.save('results/{}/{}.jpg'.format(FLAGS.name, step//FLAGS.sample_every_steps))

            if step % FLAGS.checkpoint_every_steps == 0:
                checkpoint = {
                    'G': VQGAN.G.state_dict() if FLAGS.num_gpus == 1 else VQGAN.G.module.state_dict(),
                    'D': VQGAN.D.state_dict() if FLAGS.num_gpus == 1 else VQGAN.D.module.state_dict(),
                    'GE': VQGAN.GE.state_dict() if FLAGS.num_gpus == 1 else VQGAN.GE.module.state_dict(),
                    'D_opt': VQGAN.D_opt.state_dict(),
                    'G_opt': VQGAN.G_opt.state_dict(),
                    'G_scheduler': G_scheduler.state_dict(),
                    'D_scheduler': D_scheduler.state_dict(),
                    'G_scaler': G_scaler.state_dict(),
                    'D_scaler': D_scaler.state_dict(),
                    'step': step
                }
                torch.save(checkpoint, 'results/{}/model-{}.pt'.format( FLAGS.name, step ))
            pbar.update(1)

def viz_latent():
    import seaborn as sns
    import numpy as np
    from collections import Counter

    VQGAN = LightweightVQGAN(
        latent_dim=FLAGS.latent_dim,
        image_size=FLAGS.image_size,
        downsample_size=FLAGS.downsample,
        vocab_size=FLAGS.vocab_size,
        attn_res_layers=FLAGS.attn_res_layers,
        freq_chan_attn=FLAGS.freq_chan_attn,
        perceptual_weight=FLAGS.perceptual_weight,
        enc_attn_res_layers=FLAGS.enc_attn_res_layers,
        dec_attn_res_layers=FLAGS.dec_attn_res_layers,
        ttur_mult = FLAGS.ttur_mult,
        fmap_max=FLAGS.fmap_max,
        d_fmap_max=FLAGS.d_fmap_max,
        lr=FLAGS.learning_rate,
        discriminator_iter_start=FLAGS.discriminator_iter_start,
    ).eval()
    if FLAGS.checkpoint is not None:
        state_dict = torch.load(FLAGS.checkpoint, map_location='cuda')
        VQGAN.G.load_state_dict(state_dict['G'])
        if 'GE' in state_dict:
            print('load EMA model')
            VQGAN.GE.load_state_dict(state_dict['GE'])
        else:
            VQGAN.GE.load_state_dict(state_dict['G'])

        step = state_dict['step']
    VQGAN.eval()

    sub_latent = []
    with torch.no_grad():
        for idx in tqdm(range(FLAGS.vocab_size), dynamic_ncols=True):
            latent = VQGAN.G.quantizer.embed( torch.Tensor([idx]).long().cuda() )
            sub_img = VQGAN.G.decoder(latent.view(1, FLAGS.latent_dim, 1, 1))
            sub_latent.append(sub_img.cpu())


    generated_images = torch.cat(sub_latent)

    images_grid = torchvision.utils.make_grid(generated_images, nrow = FLAGS.vocab_size//128)
    pil_image = transforms.ToPILImage()(images_grid.cpu())
    (width, height) = (pil_image.width // 4, pil_image.height // 4)
    pil_image = pil_image.resize((width, height))
    pil_image.save('results/{}/latents_{}.jpg'.format(FLAGS.name, step//FLAGS.sample_every_steps))


    if len(FLAGS.data) == 1:
        dataset = ImageDataset(FLAGS.data[0], FLAGS.image_size, aug_prob=FLAGS.dataset_aug_prob)
    else:
        print(FLAGS.data)
        dataset = ConcatDataset(
            [ 
                ImageDataset(dataset_path, FLAGS.image_size, aug_prob=FLAGS.dataset_aug_prob) \
                for dataset_path in FLAGS.data  ]
        )

    dataloader = DataLoader(dataset, 
        num_workers = FLAGS.num_workers, 
        batch_size = FLAGS.batch_size, 
        shuffle = True, drop_last = True, pin_memory = True)

    
    stats = np.zeros(( (FLAGS.vocab_size//128) * 130) )
    num_img = 0
    imgs = []
    with torch.no_grad():
        for img_batch in dataloader:
            latents = VQGAN.GE.encoder(img_batch.cuda())
            z_q, diff, ind = VQGAN.GE.quantizer(latents)
            if len(imgs) * FLAGS.batch_size <= (FLAGS.sample_grid_size)**2:
                recon_x = VQGAN.GE.decoder(z_q * VQGAN.GE.dct_weights)
                imgs.append(recon_x.cpu())

            for idx in ind.flatten():
                stats[idx.item()] += 1
    
            num_img +=1 
            if num_img > 64:
                break

    if len(imgs) > 0:
        num_rows = FLAGS.sample_grid_size
        generated_images = torch.cat(imgs)
        images_grid = torchvision.utils.make_grid(generated_images[:int(num_rows**2)], nrow = num_rows)
        pil_image = transforms.ToPILImage()(images_grid.cpu())
        pil_image.save('results/{}/viz_{}_EMA.jpg'.format(FLAGS.name, step//FLAGS.sample_every_steps))


    stats_cnt = Counter([ freq for freq in stats[:FLAGS.vocab_size]])
    print(step, stats_cnt[0], stats_cnt[0]/FLAGS.vocab_size )
    stats = stats.reshape(130, -1)
    ax = sns.heatmap(stats)
    ax.figure.savefig('results/{}/heatmap_{}.jpg'.format(FLAGS.name, step//FLAGS.sample_every_steps))



def main(argv):
    if FLAGS.viz:
        viz_latent()
    else:
        train()


if __name__ == '__main__':
  app.run(main)