import pandas as pd 
import re 
from sklearn.metrics import accuracy_score
from fitbert import FitBert



def prepare_input(x):
  '''
    # exmaple ["a clock can tell you time, so clock and time have a  ***mask*** relation",
  '''
  mask_token = "***mask***" 
  prefix = f"{x.explanation.strip('.')}, so {x.cue} and {x.association} have a "
  suffix = f" relation." 
  mask_input = prefix + mask_token + suffix 

  chunk2 = re.search(suffix, mask_input).span()
  chunk1 = re.search(f"{x.association} have a", mask_input).span()

  mask_token_position = (chunk1[1], chunk2[0])

  return pd.Series((mask_input, mask_token_position)) 


wael_amt_17rel_to_16rel_visualization = {'Synonym':'Synonym', 
'Has-Prerequisite':'Has-Prerequisite',
'Antonym':'Antonym',
'Material-MadeOf':'Made Of',
'Location': 'Location',
'PartOf':'PartOf',
'Function': 'Function', 
'Result-In': 'Result-In',
'HasProperty':'HasProperty',
'Emotion-Evaluation':'Emotion-Evaluation', 
'Time': 'Temporal', 
'Common-Phrase': 'Phrase', 
'Action': 'Action',
'Thematic':'HasContext',
'Category-Exemplar-Pairs':'Exemplar',
'Members-of-the-same-Category':'Coordinate',
}

# options = ['function', 'part of', 'location', 'antonym']
options = list(wael_amt_17rel_to_16rel_visualization.values()) 


df = pd.read_excel('data/test_wael.xlsx')
df['relation'] = df['relation'].apply(lambda x: wael_amt_17rel_to_16rel_visualization[x])
df[['fill_in_input', 'mask_token_position']] =  df.apply(lambda x: prepare_input(x), axis=1)

debug=False
#debug=True 
if debug: 
  df = df.head(10) 

# model_name="distilbert-base-uncased"
model_name = 'bert-large-uncased'
fb = FitBert(model_name=model_name)

pred_rels = []
for masked_string, mask_token_position in zip (df['fill_in_input'], df['mask_token_position']):
  # filled_in = fb.fitb(masked_string, options=options)
  filled_in_prob= fb.rank_with_prob(masked_string, options)
  # pred_rel =  filled_in[mask_token_position[0]:mask_token_position[1]]
  pred_rel = filled_in_prob[0][0]
  # print(filled_in_prob)
  # print(pred_rel, filled_in)
  pred_rels.append(pred_rel)

df['relation_pred'] = pred_rels 
acc = accuracy_score(y_true=list(df['relation']), y_pred=pred_rels) 
print(f"Acc: {acc}")
output_path = 'bert_prediction.xlsx'
df.to_excel(output_path)
print(f'save {output_path}')
df.head() 
print(df['relation_pred'])
