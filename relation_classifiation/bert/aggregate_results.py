import pandas as pd
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_column', 500)
pd.set_option('display.max_row', 200)
import copy 

def remove_unnamed_columns(df):
    return df.iloc[:, ~df.columns.str.contains('^Unnamed')]


file_prefix= './log/results_bert-base-cased_explain.log'
files  = ['./log/results_bert-base-cased_zeor-shot_-EXP.log', './log/results_bert-base-cased_zeor-shot_+EXP.log',
            './log/results_bert-base-cased_fine-tune_-EXP.log','./log/results_bert-base-cased_fine-tune_+EXP.log' ]

names = ['zeor-shot_-EXP', 'zeor-shot_+EXP', 'fine-tune_-EXP', 'fine-tune_+EXP' ]
dfs_mean = []
dfs_std = []
for file, name in zip(files, names):
    df = pd.read_csv(file)
    df = remove_unnamed_columns(df)
    print(file)
    df_mean = df.mean().to_frame().T.round(1)
    df_mean['model']  = name
    dfs_mean.append(df_mean)

    df_std = df.std().to_frame().T.round(1)
    df_std['model']  = name
    dfs_std.append(df_std)
    # print(df_mean)
    # print("")

dfs_mean = pd.concat(dfs_mean)
dfs_std = pd.concat(dfs_std)

# dfs_mean = dfs_mean.round(1)
print("MEAN")
print(dfs_mean)

print("")
print("STD")
print(dfs_std)
