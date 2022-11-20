import pandas as pd
from collections import Counter 

path = 'train_wael_augmented.json'
df = pd.read_json(path) 

df['relation'] = df['relations'].apply(lambda x: x[0]['type'])

downsample_relations = ['synonym', 'category-exemplar-pairs', 'antonym']

df_groups = []
for name, group in df.groupby(['relation']):
    if name in downsample_relations:
        group = group.sample(min(1000, len(group.index)))
    df_groups.append(group)

df_groups = pd.concat(df_groups)
output_path = 'train_wael_augment_balance.json'
print(Counter(df_groups['relation']))
df_groups = df_groups[['tokens', 'entities', 'relations', 'orig_id']]
df_groups.to_json(output_path, orient='records', indent=4)
