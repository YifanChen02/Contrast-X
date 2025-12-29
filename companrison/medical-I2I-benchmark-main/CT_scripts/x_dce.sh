





# -------------------- DCE 2D I2I -----------------------

cd /home/hao/repo/FM-translation/companrison/medical-I2I-benchmark-main/CT_scripts
conda activate progression

gpu=0

task=DCE
data_files=../../../code/data/files/2D_DCE_pair.csv

temp=/date/hao/FM_comp

target_modality=DCE1
input_modality=DCE2

output_dir=${temp}/unetflow-i2i-${input_modality}-${target_modality}
diff_ckpt=/date/hao/FM_comp/unetflow-i2i-DCE2-DCE1_stdnorm/checkpoints/checkpoint_unetflow-CTCTC-s500e_70_best.pth

# Running.....
python unetflow-dce.py --gpu $gpu  \
      --dataset_csv $data_files    \
      --batch_size 4  --n_epochs  500    --lr 1e-4 \
      --checkpoints_path  $output_dir   --input_modality  $input_modality  \
      --target_modality $target_modality  \
      --use_standard_norm   --diff_ckpt  $diff_ckpt --DEBUG  --inference









target_modality=DCE1 DCE3 
input_modality=DCE2
output_dir=${temp}/unetflow-i2i-${input_modality}-${target_modality}

python unetflow-dce.py --gpu $gpu  \
      --dataset_csv $data_files    \
      --batch_size 4  --n_epochs  500    --lr 1e-4 \
      --checkpoints_path  $output_dir   --input_modality  $input_modality  \
      --target_modality $target_modality  \
      --use_standard_norm       --DEBUG




target_modality=DCE1 
input_modality=DCE2 DCE3 
output_dir=${temp}/unetflow-i2i-${input_modality}-${target_modality}

python unetflow-dce.py --gpu $gpu  \
      --dataset_csv $data_files    \
      --batch_size 4  --n_epochs  500    --lr 1e-4 \
      --checkpoints_path  $output_dir   --input_modality  $input_modality  \
      --target_modality $target_modality  \
      --use_standard_norm        --DEBUG




# -------------------- CT 2D N2I Concat -----------------------

cd /home/hao/repo/FM-translation/companrison/medical-I2I-benchmark-main/CT_scripts
conda activate progression

gpu=6

task=CT 
data_files="../../../code/data/${task}_pair.csv"
data_files=../../../code/data/files/2D_CT_pair.csv

temp=/date/hao/FM_comp
output_dir=${temp}/unetflow-n2i-concat-${task}


python unetflow-noiset1t2.py --gpu $gpu  \
      --dataset_csv $data_files    \
      --batch_size 16  --n_epochs  500    --lr 1e-4 \
      --checkpoints_path  $output_dir    --use_standard_norm    --DEBUG
