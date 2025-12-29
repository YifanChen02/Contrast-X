





# -------------------- CT 2D I2I -----------------------

cd /home/hao/repo/FM-translation/companrison/medical-I2I-benchmark-main/CT_scripts
conda activate progression

gpu=7

task=CT 
data_files="../../../code/data/${task}_pair.csv"
data_files=../../../code/data/files/2D_CT_pair.csv

temp=/date/hao/FM_comp
output_dir=${temp}/unetflow-i2i-${task}

diff_ckpt=/date/hao/FM_comp/unetflow-i2i-CT_stdnorm/checkpoints/checkpoint_unetflow-CTCTC-s500e_218_best.pth


python unetflow-ct-ctc.py --gpu $gpu  \
      --dataset_csv $data_files    \
      --batch_size 4  --n_epochs  500    --lr 1e-4 \
      --checkpoints_path  $output_dir    --use_standard_norm      --diff_ckpt  $diff_ckpt  --DEBUG




 

 

# -------------------- CT 2D N2I Concat -----------------------

cd /home/hao/repo/FM-translation/companrison/medical-I2I-benchmark-main/CT_scripts
conda activate progression

gpu=6

task=CT 
data_files="../../../code/data/${task}_pair.csv"
data_files=../../../code/data/files/2D_CT_pair.csv

temp=/date/hao/FM_comp
output_dir=${temp}/unetflow-n2i-concat-${task}

diff_ckpt=/date/hao/FM_comp/unetflow-n2i-concat-CT_stdnorm/checkpoints/checkpoint_unetflow-CTCTC-s500e_155_best.pth

python unetflow-noise-ct-ctc.py --gpu $gpu  \
      --dataset_csv $data_files    \
      --batch_size 4  --n_epochs  500    --lr 1e-4 \
      --checkpoints_path  $output_dir    --use_standard_norm    --diff_ckpt  $diff_ckpt    --DEBUG
