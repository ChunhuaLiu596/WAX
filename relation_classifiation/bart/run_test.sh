#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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
#SBATCH -o log/220710_test_bart_large_p(r|c,a).log 
##SBATCH --gres=gpu:v100sxm2:1


# ###################################################################################################
root='/home/chunhua/CSGEN/waxrel/data/cnwae/'
# train_files=('train_wael_augmented.json')
train_files=('train_wael_1000.json')
validation_files=('dev_wael.json')
test_files=('test_wael.json' 'test_wael_mrel.json' 'test_wael_srel.json')
# test_files=('test_wael_srel.json')
# test_files=('dev_wael.json')

# run_fiilin
# run_folders=('2022-05-09/09-23-27')
# best_ckpts=("epoch\=1-step\=981")

# run rebel-large wael-1000
# run_folders=('2022-05-09/11-38-09')
# best_ckpts=("epoch\=19-step\=619")

# run bart-lage wael-1000
run_folders=('2022-07-10/15-36-48')
# ('2022-05-09/13-10-25')
best_ckpts=("epoch\=25-step\=805")
# best_ckpts=("epoch\=0-step\=490")
run_tags=run_folders


for i in "${!run_tags[@]}"; do
    train_file=$root${train_files[$i]}
    validation_file=$root${validation_files[$i]}
    # best_ckpt=${best_ckpts[$i]}
    # best_ckpt='/data/scratch/projects/punim0478/chunhua/CSGEN/waxrel/outputs/'${run_folders[$i]}'/experiments/cnwae/'${best_ckpts[$i]}'.ckpt'
    # best_ckpt="/data/scratch/projects/punim0478/chunhua/CSGEN/waxrel/outputs/2022-07-10/15-36-48/experiments/cnwae/epoch\=25-step\=805.ckpt"
    best_ckpt='/data/scratch/projects/punim0478/chunhua/CSGEN/waxrel/outputs/2022-07-10/15-36-48/experiments/cnwae/epoch\=25-step\=805.ckpt'
    echo $best_ckpt
    for j in "${!test_files[@]}"; do
        test_file=$root${test_files[$j]}

        echo ${run_tags[$i]} ${run_folders[$i]} ${test_file}
        python3 -u src/train.py model=rebel_model data=cnwae_data train=cnwae_train do_predict=True checkpoint_path=${best_ckpt} train_file=${train_file} validation_file=${validation_file} test_file=${test_file} do_train=False
        # python3 -u src/test.py model=rebel_model data=cnwae_data train=cnwae_train do_predict=True checkpoint_path=${best_ckpt} train_file=${train_file} validation_file=${validation_file} test_file=${test_file}
    done
done



