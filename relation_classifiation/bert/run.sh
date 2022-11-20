#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
## #SBATCH --partition=gpgputest
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100sxm2:1
##SBATCH --gres=gpu:A100:1
##SBATCH --constraint=dlg4
#SBATCH -o log/debug.log
##SBATCH -o log/220710_test_bert_base_cased_p(r|c,a,e).log

#
module load fosscuda/2020b 
# module load pytorch/1.9.0-python-3.8.6 
module load pytorch/1.7.1-python-3.8.6

# python BERT_Fine_Tuning_Sentence_Classification.py


# python -u bert_fine_tuning_sentence_classification.py 'zeor-shot' '+EXP' 1 
# python -u bert_fine_tuning_sentence_classification.py 'zeor-shot' '+EXP' 2 
# python -u bert_fine_tuning_sentence_classification.py 'zeor-shot' '+EXP' 3 
# python -u bert_fine_tuning_sentence_classification.py 'zeor-shot' '+EXP' 4 
# python -u bert_fine_tuning_sentence_classification.py 'zeor-shot' '+EXP' 5 

# python -u bert_fine_tuning_sentence_classification.py 'zeor-shot' '-EXP' 1 
# python -u bert_fine_tuning_sentence_classification.py 'zeor-shot' '-EXP' 2 
# python -u bert_fine_tuning_sentence_classification.py 'zeor-shot' '-EXP' 3 
# python -u bert_fine_tuning_sentence_classification.py 'zeor-shot' '-EXP' 4 
# python -u bert_fine_tuning_sentence_classification.py 'zeor-shot' '-EXP' 5 


# python -u bert_fine_tuning_sentence_classification.py 'fine-tune' '+EXP' 1 
# python -u bert_fine_tuning_sentence_classification.py 'fine-tune' '+EXP' 2 
# python -u bert_fine_tuning_sentence_classification.py 'fine-tune' '+EXP' 3 
# python -u bert_fine_tuning_sentence_classification.py 'fine-tune' '+EXP' 4 
# python -u bert_fine_tuning_sentence_classification.py 'fine-tune' '+EXP' 5 

# python -u bert_fine_tuning_sentence_classification.py 'fine-tune' '-EXP' 1 
# python -u bert_fine_tuning_sentence_classification.py 'fine-tune' '-EXP' 2 
# python -u bert_fine_tuning_sentence_classification.py 'fine-tune' '-EXP' 3 
# python -u bert_fine_tuning_sentence_classification.py 'fine-tune' '-EXP' 4 
# python -u bert_fine_tuning_sentence_classification.py 'fine-tune' '-EXP' 5 

# python -u debug.py
python -u aggregate_results.py


