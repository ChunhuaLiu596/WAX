import pandas as pd 
from tqdm import tqdm 
import re 
import json 
from sklearn.metrics import accuracy_score
from transformers import pipeline
import sys 

# filepath = './data/train_wael.json' 
# mask_type = ['subj', 'obj', 'subj-rel', 'obj-rel'] 


path = './output/filled_output_selected_relations.csv' 
df = pd.read_csv(path, header=0) 
print(f"Loading {path} with {len(df.index)} #instance")

df_original = df[['subj', 'obj', 'explanation' , 'relation']].reset_index(drop=True).drop_duplicates()
df_original['source'] = ['original']*len(df_original.index)

df_augment = df[['subj_new', 'obj_new', 'explanation_new', 'relation']].rename(columns={'subj_new': 'subj', 'obj_new': 'obj', 'explanation_new': 'explanation'}).reset_index(drop=True).drop_duplicates() 

df_augment['source'] = ['fillin']*len(df_augment.index)


df_processed = pd.concat([df_original, df_augment]).reset_index(drop=True).drop_duplicates() 
output_path = 'output/filled_output_processed.csv'
df_processed.to_csv(output_path)
print(f"Original instances: {len(df_original.index)}")
print(f"New instances: {len(df_augment.index)}")
print(f"Total instances: {len(df_processed.index)}")
print(f"save {output_path} {len(df_processed.index)} instances")

# df_filled = pd.DataFrame(filled_outputs, columns=['subj', 'obj', 'relation', 'explanation', 'mask_type', 'subj_new', 'obj_new', 'explanation_new', 'filled_sequence'])