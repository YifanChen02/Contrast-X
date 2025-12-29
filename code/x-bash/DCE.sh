
cd /home/hao/repo/FM-translation
cd /home/hao/repo/FM-translation/code/STEP1-AutoencoderModel-2D-Each



task=DCE
conda activate progression


temp=/date/hao/FM

# -------------------- Train 2DAE -----------------------
gpu=4
data_files=../data/files/2D_DCE_pair.csv
aekl_ckpt=/date/hao/FM/checkpoint/DCE/AE/ae-127-DCE.pth

output_dir=${temp}/checkpoint/DCE/2D_AE  
output_dir=${temp}/checkpoint/DCE/2D_AE_smaller  
# gpu=5

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19350 \
      train_2D_dce_autoencoder.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 6  --n_epochs  200    --lr 3e-4 \
      --cache_dir  ${temp}/cache/DCE_all      \
      --input_modality   DCE1 DCE2 DCE3    --missing_modality   DCE3  \
      --output_dir  $output_dir 



gpu=4
output_dir=${temp}/checkpoint/DCE/2D_AE_smaller_mix 
CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19750 \
      train_2D_dce_autoencoder_mix.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 6  --n_epochs  200    --lr 3e-4 \
      --cache_dir  ${temp}/cache/DCE_all      \
      --input_modality   DCE1 DCE2 DCE3    --missing_modality   DCE3  \
      --output_dir  $output_dir 




# 42 is the pure ~ 40 , 44 is the mix
CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19250 \
      train_2D_dce_autoencoder.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 6  --n_epochs  200    --lr 1e-5  \
      --cache_dir  ${temp}/cache/DCE_all     --resume  69   \
      --input_modality   DCE DCEC    --missing_modality   DCEC  \
      --output_dir  ${temp}/checkpoint/DCE/2D_AE     --use_broken      --DEBUG      




#####  ------------- Test 2DAE -----------------------
gpu=1,2,3,4
data_files=../data/files/2D_DCE_pair.csv
aekl_ckpt=/date/hao/FM/checkpoint/DCE/2D_AE_smaller/ae-71-DCE1-DCE2-DCE3.pth
aekl_ckpt=/date/hao/FM/checkpoint/DCE/2D_AE_smaller_mix/ae-24-DCE1-DCE2-DCE3.pth


data_dir=/date/yifanchen/data_no_mask/2D_jpg/Breast_split_slice_jpg
output_dir=/date/hao/FM/latent/DCE/2D

output_dir=/date/hao/FM/latent/DCE/2D_smaller

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 18270 \
      extract_2D_dce_latents.py --gpu $gpu  \
      --data_dir $data_dir  --batch_size 32   \
      --dataset_csv $data_files   --task  $task \
      --input_modality   DCE1 DCE2 DCE3    --missing_modality   DCE3  \
      --aekl_ckpt $aekl_ckpt      --output_dir  $output_dir    --DEBUG    


  




# -------------------- Train 3D AE -----------------------
cd /home/hao/repo/FM-translation
cd /home/hao/repo/FM-translation/code/STEP1-AutoencoderModel-3D

conda activate progression

task=DCE 
data_files="../data/files/${task}_pair.csv"
temp=/date/hao/FM

gpu=0,1
weight=/date/hao/FM/checkpoint/DCE/3D_AE/ae-200-DCE-DCEC.pth

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19350 \
      train_3D_dce_autoencoder.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 10  --n_epochs  500    --lr 1e-4 \
      --cache_dir  ${temp}/cache/DCE_all  \
      --input_modality   DCE DCEC    --missing_modality   DCEC  \
      --output_dir  ${temp}/checkpoint/DCE/3D_AE           --DEBUG      





CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19350 \
      train_3D_dce_autoencoder.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 4  --n_epochs  500    --lr 1e-5 \
      --cache_dir  ${temp}/cache/DCE_all    --resume 225   \
      --input_modality   DCE DCEC    --missing_modality   DCEC  \
      --output_dir  ${temp}/checkpoint/DCE/3D_AE           --DEBUG      










# -------------------- Train Flow Matching -----------------------
gpu=7
cd ~/repo/FM-translation/code/STEP1-AutoencoderModel-2D-Each/../STEP2-FlowMatching

task=DCE 
latent_dir=/date/hao/FM/latent/DCE/2D_smaller
data_dir=/date/yifanchen/data_no_mask/2D_jpg/Breast_split_slice_jpg


output_dir=/date/hao/FM/latent/DCE/FM/2D
aekl_ckpt=/date/hao/FM/checkpoint/DCE/2D_AE_smaller/ae-71-DCE1-DCE2-DCE3.pth
aekl_ckpt=/date/hao/FM/checkpoint/DCE/2D_AE_smaller_mix/ae-24-DCE1-DCE2-DCE3.pth  # Mix

data_files=../data/files/2D_DCE_pair.csv
temp=/date/hao/FM

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 \
        train_dce_2D_flowmatching_1.py   \
        --batch_size   2  \
        --dataset_csv $data_files   --task  $task \
        --gpu  $gpu      --input_modality   DCE1 DCE2 DCE3     \
        --lr    2.5e-5     \
        --n_epochs  500  \
        --grad_accum_steps 2  \
        --latent_dir $latent_dir  \
        --cache_dir ${temp}/cache/DCE_all   \
        --output_dir $output_dir \
        --aekl_ckpt  $aekl_ckpt   \
        --data_dir   $data_dir    --DEBUG




# ----------
python  latent_interpolate_ver_dce.py   \
        --batch_size   2  \
        --dataset_csv $data_files   --task  $task \
        --gpu  $gpu      --input_modality   DCE1 DCE2 DCE3     \
        --lr    2.5e-5     \
        --n_epochs  500  \
        --grad_accum_steps 2  \
        --latent_dir $latent_dir  \
        --cache_dir ${temp}/cache/DCE_all   \
        --output_dir $output_dir \
        --aekl_ckpt  $aekl_ckpt   \
        --data_dir   $data_dir    --DEBUG





task=DCE 
latent_dir=/date/hao/FM/latent/DCE/2D
data_dir=/date/yifanchen/data_no_mask/2D_jpg/Breast_split_slice_jpg


output_dir=/date/hao/FM/latent/DCE/FM/2D
aekl_ckpt=/date/hao/FM/checkpoint/DCE/2D_AE/ae-58-DCE-DCEC.pth
data_files=../data/files/2D_DCE_pair.csv
temp=/date/hao/FM


python  latent_tsne_dce.py   \
        --batch_size   2  \
        --dataset_csv $data_files   --task  $task \
        --gpu  $gpu      --input_modality   DCE DCEC    \
        --lr    2.5e-5     \
        --n_epochs  500  \
        --grad_accum_steps 2  \
        --latent_dir $latent_dir  \
        --cache_dir ${temp}/cache/DCE_all   \
        --output_dir $output_dir \
        --aekl_ckpt  $aekl_ckpt   \
        --data_dir   $data_dir 


# -------------------- Train PMRF Flow Matching -----------------------

# git clone https://github.com/ohayonguy/PMRF


latent_dir=/date/hao/FM/latent/DCE-DCEC/2D
data_dir=/date/hao/PairedContrast/DCE/low_256x256_2Dimension
output_dir=/date/hao/FM/latent/DCE-DCEC/2D
aekl_ckpt=/date/hao/FM/checkpoint/DCE/2D_AE/ae-58-DCE-DCEC.pth
data_files=../data/files/2D_DCE_pair.csv

conda activate pmrf
#!/bin/bash

python train.py \
--precision "bf16-mixed" \
--stage "flow" \
--degradation "difface" \
--train_data_root "/path/to/data/ffhq512" \
--val_data_root "/path/to/data/celebahq_test_512" \
--latent_dir  $latent_dir  \
--dataset_csv $data_files \
--aekl_ckpt   $aekl_ckpt   \
--data_dir    $data_dir  \
--arch "hdit_ImageNet256Sp4" \
--num_flow_steps 50 \
--num_gpus 16 \
--num_workers 80 \
--check_val_every_n_epoch 50 \
--train_batch_size 256 \
--val_batch_size 32 \
--img_size 128 \
--max_epochs 3850 \
--ema_decay 0.9999 \
--eps 0.0 \
--t_schedule "stratified_uniform" \
--weight_decay 1e-2 \
--lr 5e-4 \
--source_noise_std 0.1 \
--wandb_project_name "PMRF" \
--wandb_group "Blind face restoration PMRF" \
--mmse_model_arch "swinir_L" 



# \
# --mmse_model_ckpt_path "./checkpoints/swinir_restoration512_L1.pth"    # Path to the DifFace checkpoint of SwinIR



#   pip install torch_ema torchmetrics
#   pip install torchvision==0.12.0
