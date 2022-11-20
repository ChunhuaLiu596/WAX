#!/bin/bash
#BATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=64G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH -q gpgpudeeplearn
##SBATCH --gres=gpu:v100:1
#SBATCH -o log/test.log 
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100sxm2:1

#python test.py
python probe.py

# python fill_in_concept.py True
#python fill_in_concept_postprocess.py 
