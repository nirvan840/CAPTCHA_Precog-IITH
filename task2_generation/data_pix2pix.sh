#!/home/user/ocr/task2_generation/data_pix2pix.sh

cd task2_generation/pytorch-CycleGAN-and-pix2pix

conda activate pix2

python datasets/combine_A_and_B.py --fold_A /home/user/ocr/task0_dataset/data_med_new/A --fold_B /home/user/ocr/task0_dataset/data_med_new/B --fold_AB /home/user/ocr/task0_dataset/data_med_new/AB_new

# neded to install opecv 
# conda install -c conda-forge opencv