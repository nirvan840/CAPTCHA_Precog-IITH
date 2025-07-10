#!/home/user/ocr/task2_generation/train.sh

# remove old checkpoint
rm -rf /home/user/ocr/task2_generation/pytorch-CycleGAN-and-pix2pix/checkpoints/cyclegan2

# conda
conda activate pix2

# cd to pix2pix
cd /home/user/ocr/task2_generation/pytorch-CycleGAN-and-pix2pix

# Script to train a pix2pix model on your custom captcha dataset
python train.py \
  --dataroot /home/user/ocr/task0_dataset/data_med_new/AB \
  --name cyclegan2 \
  --gpu_ids 0 \
  --model cycle_gan \
  --ngf 256 \
  --ndf 128 \
  --netD n_layers \
  --n_layers_D 6 \
  --no_flip \
  --dataset_mode aligned \
  --direction AtoB \
  --num_threads 8 \
  --batch_size 1 \
  --load_size 256 \
  --crop_size 256 \
  --preprocess none \
  --max_dataset_size 10000 \
  --phase train \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --beta1 0.5 \
  --lr 0.0005 \
  --gan_mode lsgan  \
  --pool_size 50 \
  --lr_policy linear \
  --lr_decay_iters 50 \
  --display_freq 500 \
  --update_html_freq 500 \
  --display_id -1 \
  --display_port 8097 \
  --use_wandb \
  --wandb_project_name Precog-GAN 

# pix2pix
# To check
# --netD pixel
# Working kinda
# lr = 0.0001 

# visdom server for visualizing results
# python -m visdom.server
