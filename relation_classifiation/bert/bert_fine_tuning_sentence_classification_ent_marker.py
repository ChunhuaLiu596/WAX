
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

import numpy as np
from datasets import load_metric
from collections import defaultdict 
import copy

relation = ['synonym',
 'antonym',
 'action',
 'common-phrase',
 'function',
 'hasproperty',
 'result-in',
 'thematic',
 'location',
 'category-exemplar-pairs',
 'partof',
 'has-prerequisite',
 'time',
 'members-of-the-same-category',
 'material-madeof',
 'emotion-evaluation']
import pandas as pd 
rel2id = dict(zip(relation, [i for i in range(len(relation))]))
# data_files={'train': './data/wael_1560.csv', 'test': './data/wael_725.csv'}
data_files = {"train": './data/train_wael_1000.json', 
              "dev": './data/dev_wael.json', 
              "test":'./data/test_wael.json',
              'test_mrel': './data/test_wael_mrel.json',
              'test_srel': './data/test_wael_srel.json'
              }


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
  1. use original sentence 
  2. append the relation information to the sentence 
  '''

  masked_output = []
  with open(filepath) as json_file:
    f = json.load(json_file)
    for id_, row in enumerate(f):
      if id_==0:
          print("row:", row)
      relation = row['relations'][0]  
      # print(relation)

      rel = relation['type'].lower()
      subj =' '.join(row['tokens'][row['entities'][relation['head']]['start']:row['entities'][relation['head']]['end']])
      obj = ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']])
      explanaiton = " ".join(row['tokens'])

      row['tokens'][row['entities'][relation['tail']]['start']-1]= '[MASK]'
      row['tokens'][row['entities'][relation['head']]['start']-1]= '[MASK]'
      masked_output.append(( subj, obj, rel, explanaiton, " ".join(row['tokens']) ))

    df = pd.DataFrame(masked_output, columns=['head', 'tail', 'relation', 'sentence','masked_sentence' ])
    return df 


input_strategy={'entity_marker'}
pool_stregegy = {'cls', 'mention_pool'}


# Preparing input for model
text_col = 'explain'
text_col = 'pair'
for split, file in data_files.items():
  df = pd.read_json(file)
  if text_col == 'explain':
    df['text'] = df['tokens'].apply(lambda x: " ".join(x))
  elif text_col=='pair':
    df['head'] = df[['tokens', 'entities' ]].apply(lambda x: x[0][x[1][0]['start']], axis=1)
    df['tail'] = df[['tokens', 'entities' ]].apply(lambda x: x[0][x[1][1]['start']], axis=1)
    df['text'] = df[['head', 'tail']].apply(lambda x: " [SEP] ".join(x), axis=1)

  df['label'] = df['relations'].apply(lambda x: rel2id[x[0]['type'].lower()])
  df[['label','text']].to_csv(file.replace(".json", '.csv'))

dataset = load_dataset('csv', data_files={'train':'./data/train_wael_1000.csv', 
                                          'validation': './data/dev_wael.csv', 
                                          'test': './data/test_wael.csv',
                                          'test_mrel': './data/test_wael_mrel.csv',
                                          'test_srel': './data/test_wael_srel.csv'})

# for debugging
# dataset = load_dataset("yelp_review_full")
# dataset["train"][100]
# dataset["train"][10]

"""# Training the model

## Pointers:
1. f1: https://github.com/huggingface/datasets/blob/master/metrics/f1/f1.py
"""
def get_tokenizer(tokenizer):
    # tokenizer = BertTokenizer.from_pretrained(
        # 'bert-base-uncased', do_lower_case=True)
    special_tokens_dict = {'additional_special_tokens': [
        '[E1]', '[E2]', '[/E1]', '[/E2]']}  # add special token
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


# model_name = 'bert-large-cased'
model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
add_entity_marker = False
if add_entity_marker:
  tokenizer = get_tokenizer(tokenizer)



def get_entity_idx(self, _input_ids):
    e1_tks_id = self.tokenizer.convert_tokens_to_ids('[E1]')
    e2_tks_id = self.tokenizer.convert_tokens_to_ids('[E2]')
    # entity_idx = []
    pos1 = []
    pos2 = []
    for input_id in _input_ids:
        e1_idx = (input_id == e1_tks_id).nonzero().flatten().tolist()[0]
        e2_idx = (input_id == e2_tks_id).nonzero().flatten().tolist()[0]
        pos1.append(e1_idx)
        pos2.append(e2_idx)
        # entity_idx.append((e1_idx, e2_idx))
    # entity_idx = torch.Tensor(entity_idx)
    po1 = torch.Tensor(pos1)
    po2 = torch.Tensor(pos2)
    return pos1, pos2 

def compute_metrics(eval_pred):
    metric = load_metric('accuracy')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def compute_metric_extra(logits, labels, metric_name, average):
    predictions = np.argmax(logits, axis=-1)
    metric = load_metric(metric_name)
    return metric.compute(predictions=predictions, references=labels, average=average)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
# small_train_dataset = tokenized_datasets["train"]#.shuffle(seed=42).select(range(2))
# small_eval_dataset = tokenized_datasets["validation"]#.shuffle(seed=42).select(range(2))

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels= len(rel2id.keys()))

# training_args = TrainingArguments(output_dir="test_trainer")

# Train
training_args = TrainingArguments(
                    output_dir="test_trainer", 
                    evaluation_strategy="epoch", 
                    seed=42,
                    do_train=False,
                    # do_prediction=True
                    # max_steps=1
                    )
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# if zero-shot, comment this line
# trainer.train()


# Test
df_results = []
for split in ['test', 'test_mrel', 'test_srel']:
  print(f"############## {split} ######")
  test_dataset = tokenized_datasets[split]
  outputs = trainer.predict(test_dataset) 

  metrics = outputs[-1]
  for metric in ['precision', 'recall', 'f1']:
    score = compute_metric_extra(outputs[0], outputs[1], metric, average='macro')
    # print("{} : {}".format(metric, score))
    metrics.update(score)
  
  df_results.append({'split': split, 
                     'P': round(metrics['precision'],3),
                     'R': round(metrics['recall'],3),
                     'F1': round(metrics['f1'], 3),
                    'Acc': round(metrics['test_accuracy'], 3)}
                  )
  
df_results = pd.DataFrame.from_dict(df_results)



# df_results.head()

output_path = f'log/results_{model_name}_{text_col}.csv'
df_results.to_csv(output_path)
print(df_results)
df_results['P'] = df_results['P'].mul(100)
df_results['R'] = df_results['R'].mul(100)
df_results['F1'] = df_results['F1'].mul(100)
df_results['Acc'] = df_results['Acc'].mul(100)
print(df_results)
print(f"save {output_path}")