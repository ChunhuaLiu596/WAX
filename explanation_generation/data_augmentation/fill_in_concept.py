import pandas as pd 
from tqdm import tqdm 
import re 
import json 
from sklearn.metrics import accuracy_score
from transformers import pipeline
import sys 
from stopwords import STOP_WORDS 

filepath = 'data/train_wael.json' 
mask_type = ['subj', 'obj', 'subj-rel', 'obj-rel'] 

wael_amt_17rel_to_16rel_visualization = {'Synonym':'Synonym', 
'Has-Prerequisite':'Has-Prerequisite',
'Antonym':'Antonym',
'Material-MadeOf':'Made Of',
'Location': 'Location',
'PartOf':'PartOf',
'Function': 'Function', 
'Result-In': 'Result-In',
'HasProperty':'Has-Property',
'Emotion-Evaluation':'Emotion-Evaluation', 
'Time': 'Temporal', 
'Common-Phrase': 'Phrase', 
'Action': 'Action',
'Thematic':'Has-Context',
'Category-Exemplar-Pairs':'Exemplar',
'Members-of-the-same-Category':'Coordinate',
}

# selected_relations = ['Result-In', 'Emotion-Evaluation', 'Time', 'Category-Exemplar-Pairs', 'Members-of-the-same-Category']
# selected_relations = ['Result-In']
selected_relations  = wael_amt_17rel_to_16rel_visualization.keys() 

def mask_concepts(filepath, add_relation=False, select_relation=True): 
  '''
    # exmaple: 
    masked_strings = ["a clock can tell you [MASK] .", 
    "a clock can tell you [MASK] so they have a function relation .", 
    "a [MASK]  can tell you time so they have a function relation.", ]
  ]
  Two strategies can be used to mask the input: 
  1. mask the cue 
  2. mask the association
  
  Two strategy for prefix:
  1. use original explanation 
  2. append the relation information to the explanation 
  '''

  masked_output = []
  with open(filepath) as json_file:
    f = json.load(json_file)
    for id_, row in enumerate(f):
      if id_==0:
          print("row:", row)
      relation = row['relations'][0]  
      # print(relation)
      if select_relation and relation['type'] not in selected_relations: continue  

      rel = relation['type'].lower()
      subj =' '.join(row['tokens'][row['entities'][relation['head']]['start']:row['entities'][relation['head']]['end']])
      obj = ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']])
      explanaiton = " ".join(row['tokens'])

      subj_masked = row['tokens'].copy()
      subj_masked[row['entities'][relation['head']]['start']]= '[MASK]'

      obj_masked = row['tokens'].copy()
      obj_masked[row['entities'][relation['tail']]['start']]= '[MASK]'

      token_str_len = len(" ".join(row['tokens']))
      masked_output.append((" ".join(subj_masked), subj, obj, rel, explanaiton, 'subj', (token_str_len, token_str_len)))
      masked_output.append((" ".join(obj_masked), subj, obj, rel, explanaiton, 'obj', (token_str_len, token_str_len)))
      

      if add_relation: 
        rel_suffix_start_subj, rel_suffix_start_obj = [] , []
        rel_readable_string = wael_amt_17rel_to_16rel_visualization[relation['type']] 

        suffix_rel = f", so they have a {rel_readable_string } relation."
        subj_masked = " ".join(subj_masked).strip(".")
        obj_masked = " ".join(obj_masked).strip(".")

        rel_suffix_start_subj.append(len(subj_masked))
        rel_suffix_start_obj.append(len(obj_masked))

        subj_masked = subj_masked + suffix_rel
        obj_masked = obj_masked + suffix_rel 

        rel_suffix_start_subj.append(len(subj_masked))
        rel_suffix_start_obj.append(len(obj_masked))
        # rel_suffix_start_subj = re.search(suffix_rel, subj_masked).span()
        # rel_suffix_start_obj = re.search(suffix_rel, subj_masked).span()

        masked_output.append((subj_masked, subj, obj, rel, explanaiton, 'subj-rel', rel_suffix_start_subj ))
        masked_output.append((obj_masked, subj, obj, rel, explanaiton, 'obj-rel', rel_suffix_start_obj))

      
      # print(row['tokens'])
      # print(subj_masked)
      # print(obj_masked)
      # print("-"*70)
    masked_output = pd.DataFrame(masked_output, columns=['masked_input', 'subj', 'obj', 'relation', 'explanation', 'mask_type', 'suffix_position' ]) 
    return masked_output 
      
      # print(subj, obj, rel)



# options = ['function', 'part of', 'location', 'antonym']
options = list(wael_amt_17rel_to_16rel_visualization.values()) 

df = mask_concepts(filepath, add_relation=True) 
print(f"#instances {len(df.index)}")
# df['relation'] = df['relation'].apply(lambda x: wael_amt_17rel_to_16rel_visualization[x])
# debug=False
# debug=True 
debug = eval(sys.argv[1])
if debug=="True": 
  df = df.head(10)

# len(df.index)
df.mask_type.head()


filled_outputs = []
top_k=5
if len(selected_relations) >0: 
  top_k=10

unmasker = pipeline('fill-mask', model='bert-large-uncased')
# for masked_string in df['masked_input']: 
for i, row in enumerate(tqdm(df.itertuples(), total=len(df.index))):
  if i %10 == 0: print(row.masked_input) 
  outputs = unmasker(row.masked_input, top_k=top_k)  
  # print(filled_string)
  for output in outputs: 
    filled_token = output['token_str']

    #####Add conditions to filter unwanted ################
    # filter the repetation of a concept in the explanation. See the the following example
    # [MASK] is the capability to do a particular job . -> capacity 
    if not filled_token.isalpha():  continue
    if filled_token in STOP_WORDS: continue 
    if len(filled_token)<=1: continue 
    if filled_token in row.masked_input.split(): continue
    if filled_token.startswith("#"): continue
    #####Add conditions to filter unwanted ################

    # if filled_token not in row.masked_input.split():
    if i %10 == 0: 
      print("{}\t {}".format(output['token_str'], row.masked_input))
    # print("{}\t{}\t{}".format(output['token_str'], output['sequence'], round(output['score'],4)))
    # print(output)
    if row.mask_type in ['subj', 'subj-rel']: 
      new_subj = filled_token
      new_obj = row.obj  
      new_explanation = output['sequence']
      if row.mask_type == 'subj-rel': 
        rel_suffix_start_subj = re.search(", so they have a", output['sequence']).span()[0] 
        new_explanation = output['sequence'][:rel_suffix_start_subj]

    elif row.mask_type in ['obj', 'obj-rel']:
      new_obj = filled_token 
      new_subj = row.subj  
      new_explanation = output['sequence']
      if row.mask_type == 'obj-rel':
          rel_suffix_start_obj = re.search(", so they have a", output['sequence']).span()[0] 
          new_explanation = output['sequence'][:rel_suffix_start_obj]

      filled_outputs.append((row.subj, row.obj, row.relation, row.explanation, row.mask_type, new_subj,new_obj, new_explanation,output['sequence'])) 
  print('-'*70)

df_filled = pd.DataFrame(filled_outputs, columns=['subj', 'obj', 'relation', 'explanation', 'mask_type', 'subj_new', 'obj_new', 'explanation_new', 'filled_sequence'])

output_path = 'output/filled_output_selected_relations.csv'
# df_filled.to_excel(output_path) 
df_filled.to_csv(output_path) 
print(f"save {df_filled}")
df_filled.head()
