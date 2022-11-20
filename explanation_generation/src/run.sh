#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem=64G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH -q gpgpudeeplearn
##SBATCH --gres=gpu:v100:3
#SBATCH -o ../log/debug.log 
#SBATCH --gres=gpu:v100sxm2:3


# run demo
python src/transformer_demo.py 

# run nyt
# python train.py model=rebel_model data=conll04_data train=conll04_train