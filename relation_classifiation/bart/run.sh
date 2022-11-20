#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=100:00:00
#SBATCH --mem=64G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH -q gpgpudeeplearn
##SBATCH --gres=gpu:v100:3
#SBATCH --gres=gpu:v100sxm2:3
##SBATCH --gres=gpu:A100:1
##SBATCH --constraint=dlg4
#SBATCH -o log/220327_cnwae_2.log 


# run demo
# python src/transformer_demo.py 

# test gpu 
# python3 test_gpu.py

# run conll
# python3 src/train.py model=rebel_model data=conll04_data train=conll04_train

# run  cnwae, cnwae_train is linked to conll04_train
# python3 -u src/train.py model=rebel_model data=conll04_data train=conll04_train

echo "##### without DEV (early stopping) ####"
echo 'CNREL 32788 	None'
rm data/cnwae/*.cache
python3 -u src/train.py model=rebel_model data=cnwae_data train=cnwae_train train_file='/home/chunhua/CSGEN/rebel/data/cnwae/train_cn.json' max_steps=1000

echo 'CNREL 32788 + WAE 900	None'
rm data/cnwae/*.cache
python3 -u src/train.py model=rebel_model data=cnwae_data train=cnwae_train train_file='/home/chunhua/CSGEN/rebel/data/cnwae/train_cnwae.json' max_steps=1000

echo 'WAE900	None'
rm data/cnwae/*.cache
python3 -u src/train.py model=rebel_model data=cnwae_data train=cnwae_train train_file='/home/chunhua/CSGEN/rebel/data/cnwae/dev.json' max_steps=1000


echo "##### with DEV (early stopping) ####"
echo 'CNREL 32788 	WAE900'
rm data/cnwae/*.cache
python3 -u src/train.py model=rebel_model data=cnwae_data train=cnwae_train train_file='/home/chunhua/CSGEN/rebel/data/cnwae/train_cn.json' max_steps=20000 apply_early_stopping=True 

echo 'CNREL 32788 + WAE 900	WAE900  '
rm data/cnwae/*.cache
python3 -u src/train.py model=rebel_model data=cnwae_data train=cnwae_train train_file='/home/chunhua/CSGEN/rebel/data/cnwae/train_cnwae.json' max_steps=20000 apply_early_stopping=True 

echo 'WAE900	WAE900'
rm data/cnwae/*.cache
python3 -u src/train.py model=rebel_model data=cnwae_data train=cnwae_train train_file='/home/chunhua/CSGEN/rebel/data/cnwae/dev.json' max_steps=20000 apply_early_stopping=True 


#  apply_early_stopping=True 
# python3 -u src/train.py model=default_model data=cnwae_data train=cnwae_train apply_early_stopping=True 
# train_batch_size=2 learning_rate=0.00002



