#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=100:00:00
#SBATCH --mem=64G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
##SBATCH --gres=gpu:v100sxm2:1
##SBATCH --gres=gpu:A100:1
##SBATCH --constraint=dlg4
#SBATCH -o log/220516_train_wael_aug_balance_T5_large_test.log
##SBATCH -o log/debug.log


export WANDB_START_METHOD=fork
root='/home/chunhua/CSGEN/wax/data/cnwae/'

# train_files=('train_wael_augmented.json')
# train_files=('train_wael_1000.json')
train_files=('train_wael_augment_balance.json')
#  'train_cn_25217.json' 'train_cnwae_26117.json')
validation_files=('dev_wael.json')
test_files=('test_wael.json')
#  'dev_900.json' 'dev_900.json')
max_steps=2000
train_batch_sizes=(4)
#  6 4)

echo "##### max_steps=${max_steps} ####"

for i in "${!train_files[@]}"; do
    train_file=$root${train_files[$i]}
    validation_file=$root${validation_files[$i]}
    test_file=$root${test_files[$i]}
    rm -f data/cnwae/*.cache

    for j in "${!train_batch_sizes[@]}"; do
        echo 'train_file='${train_file}, 'validation_file='${validation_file} 'batch_size='${train_batch_sizes[$j]} "learning_rate=0.00002"
        # CUDA_VISIBLE_DEVICES=0,1,2,3 
        python3 -u src/train.py model=default_model data=cnwae_data train=cnwae_train train_file=${train_file} validation_file=${validation_file}  max_steps=${max_steps} train_batch_size=${train_batch_sizes[$j]} learning_rate=0.00002 do_predict=True  test_file=${test_file} model_name_or_path='t5-small' config_name='t5-small' tokenizer_name='t5-small'
    done
done

######################################################## ########################################################
# for i in "${!train_files[@]}"; do
#     train_file=$root${train_files[$i]}
#     validation_file=$root${validation_files[$i]}
#     rm -f data/cnwae/*.cache

#     for j in "${!train_batch_sizes[@]}"; do
#         echo 'train_file='${train_file}, 'validation_file='${validation_file} 'batch_size='${train_batch_sizes[$j]} "learning_rate=0.00001"

#         python3 -u src/train.py model=rebel_model data=cnwae_data train=cnwae_train train_file=${train_file} validation_file=${validation_file}  max_steps=${max_steps} train_batch_size=${train_batch_sizes[$j]} learning_rate=0.00001
#     done
# done
