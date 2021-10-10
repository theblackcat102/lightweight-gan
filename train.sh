CUDA_VISIBLE_DEVICES=1 python -m examples.vq_vae \
    --batch_size 3 \
    --gradient_accumulate_every 10 \
    --image_size 512 \
    --downsample 32 \
    --init_temp 2.0 \
    --aug_prob 0.0 \
    --recon_weight 1 \
    --num_gpus 1 \
    --perceptual_weight -1 \
    --name vqgan_32_attn \
    --learning_rate 2e-4 \
    --dec_attn_res_layers 16 \
    --dec_attn_res_layers 32 \
    --dec_attn_res_layers 64 \
    --smooth_l1_loss --vocab_size 18384 \
    --sample_grid_size 5  \
    --discriminator_iter_start 10000 \
    --checkpoint results/vqgan_32_attn/model-12000.pt \
    --data ./sample_images



CUDA_VISIBLE_DEVICES=1,2 python -m examples.vq_vae \
    --batch_size 32 \
    --gradient_accumulate_every 2 \
    --image_size 256 \
    --downsample 16 \
    --init_temp 2.0 \
    --aug_prob 0.0 \
    --recon_only \
    --recon_weight 1 \
    --fmap_max 512 \
    --d_fmap_max 256 \
    --num_gpus 2 \
    --perceptual_weight -1 \
    --name vqgan_small_attn \
    --learning_rate 2e-4 \
    --dec_attn_res_layers 16 \
    --dec_attn_res_layers 32 \
    --smooth_l1_loss --vocab_size 18384 \
    --sample_grid_size 5  \
    --discriminator_iter_start 10000 \
    --data ./sample_images
