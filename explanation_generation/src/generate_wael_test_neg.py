import json
import logging
import pandas as pd
import math
from collections import defaultdict
import copy

_DESCRIPTION = """\
    CNWAE is made of sentences from news articles, annotated with four entity types (person, organization, location, other)
    and five relation types (kill, work for, organization based in, live in, located in).
"""

_URL = ""
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "val.json",
    "test": _URL + "test.json",
}

# mapping = {'Kill': 'killed by', 'Live_In': 'residence', 'Located_In': 'location', 'OrgBased_In': 'headquarters location', 'Work_For': 'employer'}
mapping_types  = {'NOUN': '<noun>','VERB': '<verb>','CCONJ': '<cconj>', 'AUX': '<aux>',  'PROPN': '<propn>', 'X': '<x>', 'ADJ': '<adj>', 'DET': '<det>', 'PRON': '<pron>', 'ADV': '<adv>', 'ADP': '<adp>', 'SCONJ': '<sconj>', 'INTJ': '<intj>', 'NUM': '<num>'}
# from utils_relation_mapping import rel2
# mapping  ={'action':'action', 'category-exemplar-pairs': 'category exemplar pairs', 'causes': 'causes', 'hasprerequisite': 'has prerequisite', 'hasproperty': 'has property', 'location': 'location', 'partof': 'part of', 'synonym': 'synonym', 'thematic': 'thematic', 'usedfor': 'used for', 'antonym': 'antonym', 'common-phrase': 'common phrase'}
# mapping={'atlocation': 'at location',
#  'partof': 'part of',
#  'hasa': 'hasa',
#  'hassubevent': 'has subevent',
#  'causes': 'causes',
#  'hasproperty': 'has property',
#  'mannerof': 'manner of',
#  'isa': 'isa',
#  'usedfor': 'used for',
#  'synonym': 'synonym',
#  'antonym': 'antonym',
#  'hasprerequisite': 'has prerequisite',
#  'hascontext': 'has context',
#  'capableof': 'capable of',
#  'receivesaction': 'receives action'}

mapping = {'synonym': 'synonym',
'antonym': 'antonym',
'action': 'action',
'common-phrase': 'common phrase',
'function': 'function',
'hasproperty': 'property',
'result-in': 'result in',
'thematic': 'thematic',
'location': 'location',
'category-exemplar-pairs': 'exemplar pairs',
'partof': 'part of',
'has-prerequisite': 'has prerequisite',
'time': 'time',
'members-of-the-same-category': 'same category',
'material-madeof': 'made of',
'emotion-evaluation': 'emotion evaluation'}


def generate_examples( filepath, add_neg_rel=False):
    """This function returns the examples in the raw (text) triplet form."""
    logging.info("generating examples from = %s", filepath)
    if isinstance(filepath, list):
        filepath = filepath[0]

    print(f"filepath {filepath} ")
    examples = []
    flag = 0
    with open(filepath) as json_file:
        f = json.load(json_file)
        for id_, row in enumerate(f):
            if id_==0:
                print("row:", row)
            # if id_<5:
            relation = row['relations'][0]
            explain = ' '.join(row['tokens'])
            subj = ' '.join(row['tokens'][row['entities'][relation['head']]['start']:row['entities'][relation['head']]['end']])
            obj = ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']])
            relation_mapped = relation['type']

            for rel in mapping.keys(): 
                cur_row = copy.deepcopy(row) 
                cur_row['relations'][0]['type'] = rel
                if rel!=relation_mapped.lower():
                    cur_row['relations'][0]['negative_example'] = 1 
                else:
                    cur_row['relations'][0]['negative_example'] = 0 

                example= {
                        "tokens": row['tokens'],
                        "entities" : row['entities'],
                        "relations" : cur_row['relations'],
                        "orig_id": row["orig_id"],
                        "gold_rel": relation_mapped,
                        "prompt_rel": rel,
                        "subj": subj,
                        "obj": obj,
                    }
                # yield example 
                examples.append(example)
        output_path = filepath.replace('.json', '_neg.json')

        with open(output_path, 'w') as fout:
            json.dump(examples, fout, indent=4)
        print(f"save {output_path}")


# if __name__=='__main__':
# filepath = '../data/cnwae/test_wael.json'
# filepath = '../data/wae_v2/rel_diversity/bart/test_wael.json'
# filepath = '../data/cnwae/test_debug.json'
filepath = '../data/cnwae/test_wael.json'
print("input filepath", filepath)
generate_examples(filepath=filepath, add_neg_rel=True) 

# output_path = filepath.replace('.json', '_neg.json')
# df = pd.read_json(output_path)
# df['negative_examples'] = df['relations'].apply(lambda x: x[0]['negative_example'])
# print(len(df.query("negative_examples == False").index))