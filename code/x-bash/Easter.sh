
cd /home/hao/repo/FM-translation
cd /home/hao/repo/FM-translation/code/STEP1-AutoencoderModel-2D




conda activate progression

task=CT 
data_files="../data/${task}_pair.csv"
temp=/date/hao/FM

# -------------------- Train 2DAE -----------------------
gpu=0,1,2,3,4
data_files=../data/files/2D_CT_pair.csv
aekl_ckpt=/date/hao/FM/checkpoint/CT/AE/ae-127-CT-CTC.pth


CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19250 \
      train_2D_ct_autoencoder.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 18  --n_epochs  200    --lr 3e-4 \
      --cache_dir  ${temp}/cache/CT_all      \
      --input_modality   CT CTC    --missing_modality   CTC  \
      --output_dir  ${temp}/checkpoint/CT/2D_AE  

      

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19250 \
      train_2D_ct_autoencoder.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 18  --n_epochs  200    --lr 3e-4 \
      --cache_dir  ${temp}/cache/CT_all      \
      --input_modality   CT CTC    --missing_modality   CTC  \
      --output_dir  ${temp}/checkpoint/CT/2D_AE      --use_broken        --DEBUG      




CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19250 \
      train_2D_ct_autoencoder.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 18  --n_epochs  200    --lr 3e-4 \
      --cache_dir  ${temp}/cache/CT_all     --resume  27  \
      --input_modality   CT CTC    --missing_modality   CTC  \
      --output_dir  ${temp}/checkpoint/CT/2D_AE     --use_broken      --DEBUG      




#####  Test 2DAE
gpu=6
data_files=../data/files/2D_CT_pair.csv
aekl_ckpt=/date/hao/FM/checkpoint/CT/AE/ae-132-CT-CTC.pth

data_dir=/date/hao/PairedContrast/CT/low_256x256_2Dimension
output_dir=/date/hao/FM/latent/CT/2D

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 18250 \
      extract_2D_latents.py --gpu $gpu  \
      --data_dir $data_dir  \
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
gpu=0,1

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 \
        train_flowmatching.py   \
        --batch_size   4  \
        --gpu  $gpu      --input_modality  t1n   \
        --lr    2.5e-5     \
        --n_epochs  500  \
        --grad_accum_steps 3  \
        --latent_dir $temp_dir/Tumor/AE/results/brats_latent/t1n  \
        --cache_dir  $temp_dir/Tumor/AE/results/cache_dm_t1n   \
        --output_dir $temp_dir/Tumor/AE/results/output \
        --aekl_ckpt  $temp_dir/Tumor/AE/results/output/ae-50-t1n.pth   \
        --data_dir   ~/hao/data/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData     --DEBUG








