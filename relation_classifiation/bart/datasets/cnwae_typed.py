import json
import logging

import datasets

import math
from collections import defaultdict

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


# maping= {'synonym': '<synonym>',
#              'antonym': '<antonym>',
#              'action': '<action>',
#              'common-phrase': '<common phrase>',
#              'function': '<function>',
#              'hasproperty': '<property>',
#              'result-in': '<result in>',
#              'thematic': '<thematic>',
#              'location': '<location>',
#              'category-exemplar-pairs': '<exemplar pairs>',
#              'partof': '<part of>',
#              'has-prerequisite': '<has prerequisite>',
#              'time': '<time>',
#              'members-of-the-same-category': '<same category>',
#              'material-madeof': '<made of>',
#              'emotion-evaluation': '<emotion evaluation>'}


class CNWAEConfig(datasets.BuilderConfig):
    """BuilderConfig for CNWAE."""

    def __init__(self, **kwargs):
        """BuilderConfig for CNWAE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CNWAEConfig, self).__init__(**kwargs)


class CNWAE(datasets.GeneratorBasedBuilder):
    """CNWAE"""

    BUILDER_CONFIGS = [
        CNWAEConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "triplets": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            # homepage="https://www.aclweb.org/anthology/W04-2401/",
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = {
                "train": self.config.data_files["train"], # self.config.data_dir + "en_train.jsonl",
                "dev": self.config.data_files["dev"], #self.config.data_dir + "en_val.jsonl",
                "test": self.config.data_files["test"], #self.config.data_dir + "en_test.jsonl",
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]


    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) triplet form."""
        logging.info("generating examples from = %s", filepath)
        if isinstance(filepath, list):
            filepath = filepath[0]

        print(f"filepath {filepath} ")
        with open(filepath) as json_file:
            f = json.load(json_file)
            for id_, row in enumerate(f):
                if id_==0:
                    print("row:", row)
                relation = row['relations'][0]

                #1. input: cue + association 
                subj_word = ' '.join(row['tokens'][row['entities'][relation['head']]['start']:row['entities'][relation['head']]['end']]) 
                obj_word = ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']]) 
                subj_pos = f' {mapping_types[row["entities"][relation["head"]]["type"]]} '
                obj_pos = f' {mapping_types[row["entities"][relation["tail"]]["type"]]} '

                #1. only subj - obj 
                # source = ' <subj> ' + subj_word + ' <obj> ' + obj_word  

                #2. only subj , obj , pos 
                # source = ' <subj> ' + subj_word + sujb_pos + ' <obj> ' + obj_word + obj_pos

                #3. input: exp + cue + association 
                source = ' '.join(row['tokens']) + ' <subj> ' + subj_word + subj_pos + ' <obj> ' + obj_word + obj_pos

                target ='<triplet> ' +  ' '.join(row['tokens'][row['entities'][relation['head']]['start']:row['entities'][relation['head']]['end']]) + ' <subj> '\
                + ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']]) + ' <obj> '\
                + mapping[relation['type'].lower()]
                print("source:", source)
                print("target:", target)
                # return str(row["orig_id"]), {
                yield str(row["orig_id"]), {
                    "title": str(row["orig_id"]),
                    "context": source,
                    "id": str(row["orig_id"]),
                    "triplets": target,
                    }
      

    # def _generate_examples(self, filepath):
    #     # """This function returns the examples in the raw (text) triplet form."""
    #     # logging.info("generating examples from = %s", filepath)
    #     if isinstance(filepath, list):
    #         filepath = filepath[0]

    #     print(f"filepath {filepath} ")
    #     with open(filepath) as json_file:
    #         f = json.load(json_file)
    #         for id_, row in enumerate(f):
    #             if id_==0:
    #                 print("row:", row)
    #             triplets = ''
    #             prev_head = None
    #             for relation in row['relations']:
    #                 if prev_head == relation['head']:
    #                     triplets += f' {mapping_types[row["entities"][relation["head"]]["type"]]} ' + ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']]) + f' {mapping_types[row["entities"][relation["tail"]]["type"]]} ' + mapping[relation['type']]
    #                 elif prev_head == None:
    #                     triplets += '<triplet> ' + ' '.join(row['tokens'][row['entities'][relation['head']]['start']:row['entities'][relation['head']]['end']]) + f' {mapping_types[row["entities"][relation["head"]]["type"]]} ' + ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']]) + f' {mapping_types[row["entities"][relation["tail"]]["type"]]} ' + mapping[relation['type']]
    #                     prev_head = relation['head']
    #                 else:
    #                     triplets += ' <triplet> ' + ' '.join(row['tokens'][row['entities'][relation['head']]['start']:row['entities'][relation['head']]['end']]) + f' {mapping_types[row["entities"][relation["head"]]["type"]]} ' + ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']]) + f' {mapping_types[row["entities"][relation["tail"]]["type"]]} ' + mapping[relation['type']]
    #                     prev_head = relation['head']
    #             text = ' '.join(row['tokens'])

    #             # return str(row["orig_id"]), {
    #             yield str(row["orig_id"]), {
    #                 "title": str(row["orig_id"]),
    #                 "context": text,
    #                 "id": str(row["orig_id"]),
    #                 "triplets": triplets,
    #                 }