
cd /home/hao/repo/FM-translation
cd /home/hao/repo/FM-translation/code/STEP1-AutoencoderModel-2D




conda activate progression

task=CT 
data_files="../data/${task}_pair.csv"
temp=/date/hao/FM

# -------------------- Train 2DAE -----------------------
cd /home/hao/repo/FM-translation/code/STEP1-AutoencoderModel-2D-Each
gpu=1,2,3,4
gpu=5,6,7
data_files=../data/files/2D_CT_pair.csv
aekl_ckpt=/date/hao/FM/checkpoint/CT/AE/ae-127-CT-CTC.pth

output_dir=${temp}/checkpoint/CT/2D_AE  
output_dir=${temp}/checkpoint/CT/2D_AE_smaller_8

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19360 \
      train_2D_ct_autoencoder.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 12  --n_epochs  200    --lr 3e-4 \
      --cache_dir  ${temp}/cache/CT_all     \
      --input_modality   CT CTC    --missing_modality   CTC  \
      --output_dir  $output_dir  --resume 45



# 42 is the pure ~ 40 , 44 is the mix
CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19250 \
      train_2D_ct_autoencoder.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 6  --n_epochs  200    --lr 1e-5  \
      --cache_dir  ${temp}/cache/CT_all     --resume  69   \
      --input_modality   CT CTC    --missing_modality   CTC  \
      --output_dir  ${temp}/checkpoint/CT/2D_AE     --use_broken      --DEBUG      




#####  ------------- Test 2DAE -----------------------
gpu=5,6,7
data_files=../data/files/2D_CT_pair.csv
aekl_ckpt=/date/hao/FM/checkpoint/CT/2D_AE_smaller/ae-62-CT-CTC.pth

data_dir=/date/hao/PairedContrast/CT/low_256x256_2Dimension
output_dir=/date/hao/FM/latent/CT-CTC/2D



CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 18250 \
      extract_2D_ct_latents.py --gpu $gpu  \
      --data_dir $data_dir  --batch_size 32   \
      --dataset_csv $data_files   --task  $task \
      --input_modality   CT CTC    --missing_modality   CTC  \
      --aekl_ckpt $aekl_ckpt      --output_dir  $output_dir    --DEBUG    


  




# -------------------- Train 3D AE -----------------------
cd /home/hao/repo/FM-translation
cd /home/hao/repo/FM-translation/code/STEP1-AutoencoderModel-3D

conda activate progression

task=CT 
data_files="../data/files/${task}_pair.csv"
temp=/date/hao/FM

gpu=0,1
weight=/date/hao/FM/checkpoint/CT/3D_AE/ae-200-CT-CTC.pth

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19350 \
      train_3D_ct_autoencoder.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 10  --n_epochs  500    --lr 1e-4 \
      --cache_dir  ${temp}/cache/CT_all  \
      --input_modality   CT CTC    --missing_modality   CTC  \
      --output_dir  ${temp}/checkpoint/CT/3D_AE           --DEBUG      





CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19350 \
      train_3D_ct_autoencoder.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 4  --n_epochs  500    --lr 1e-5 \
      --cache_dir  ${temp}/cache/CT_all    --resume 225   \
      --input_modality   CT CTC    --missing_modality   CTC  \
      --output_dir  ${temp}/checkpoint/CT/3D_AE           --DEBUG      






# -------------------- Train Flow Matching -----------------------
gpu=4,5,6,7
cd ../STEP2-FlowMatching

task=CT 
latent_dir=/date/hao/FM/latent/CT-CTC/2D
data_dir=/date/hao/PairedContrast/CT/low_256x256_2Dimension
output_dir=/date/hao/FM/latent/CT-CTC/2D
aekl_ckpt=/date/hao/FM/checkpoint/CT/2D_AE_smaller/ae-62-CT-CTC.pth
data_files=../data/files/2D_CT_pair.csv
temp=/date/hao/FM

fm_ckpt=/date/hao/FM/latent/CT-CTC/2D/dm-unet-ep-15-CT-CTC-Adrenal.pth

# 2.5e-5

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 \
        train_ct_2D_flowmatching.py   \
        --batch_size   2  \
        --dataset_csv $data_files   --task  $task \
        --gpu  $gpu      --input_modality   CT CTC    \
        --lr    3e-4     \
        --n_epochs  500  \
        --grad_accum_steps 2  \
        --latent_dir $latent_dir  \
        --cache_dir ${temp}/cache/CT_all   \
        --output_dir $output_dir \
        --aekl_ckpt  $aekl_ckpt   \
        --diff_ckpt $fm_ckpt   \
        --data_dir   $data_dir      --DEBUG




python simple_fm.py \
    --lr 3e-4 \
    --lr_min 1e-6 \
    --batchsize 6 \
    --epochs 300 \
    --num_workers 4

    

# -------------------- Train PMRF Flow Matching -----------------------

# git clone https://github.com/ohayonguy/PMRF


latent_dir=/date/hao/FM/latent/CT-CTC/2D
data_dir=/date/hao/PairedContrast/CT/low_256x256_2Dimension
output_dir=/date/hao/FM/latent/CT-CTC/2D
aekl_ckpt=/date/hao/FM/checkpoint/CT/2D_AE/ae-58-CT-CTC.pth
data_files=../data/files/2D_CT_pair.csv

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
--img_size 512 \
--max_epochs 3850 \
--ema_decay 0.9999 \
--eps 0.0 \
--t_schedule "stratified_uniform" \
--weight_decay 1e-2 \
--lr 5e-4 \
--source_noise_std 0.1 \
--wandb_project_name "PMRF" \
--wandb_group "Blind face restoration PMRF" \
--mmse_model_arch "swinir_L" \
--mmse_model_ckpt_path "./checkpoints/swinir_restoration512_L1.pth"    # Path to the DifFace checkpoint of SwinIR



#   pip install torch_ema torchmetrics
#   pip install torchvision==0.12.0
