import numpy as np
import pandas as pd
import torch
import json
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_metric

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

 

def compute_metrics(preds, labels, metric_name='rouge'):
    # metric_name = "rouge" # if hparams.task.startswith("summarization") else "sacrebleu"
    metric = load_metric(metric_name)
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        if metric_name == "rouge":
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        elif metric_name in ['bleu', 'sacrebleu']:
            preds = [nltk.word_tokenize(pred) for pred in preds]
            labels = [nltk.word_tokenize(label) for label in labels]
            labels = [[label] for label in labels]
        else:  # sacrebleu
            labels = [[label] for label in labels]

        return preds, labels
    # # preds, labels = eval_preds
    # if isinstance(preds, tuple):
    #     preds = preds[0]
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # if hparams.ignore_pad_token_for_loss:
    #     # Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # # Some simple post-processing
    decoded_preds = preds
    decoded_labels = labels 

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    if metric_name == "rouge":
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    else:
        result_tmp = metric.compute(predictions=decoded_preds, references=decoded_labels)
        # result = {"bleu": result["score"]}
        result = {"bleu": result_tmp["bleu"], "bleu-3": result_tmp["precisions"][2], "bleu-4": result_tmp['precisions'][3]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result




predictions = [
    ["hello", "there", "general", "kenobi"],                             # tokenized prediction of the first sample
    ["foo", "bar", "foobar"]                                             # tokenized prediction of the second sample
]
# references = [
#     [["hello", "there", "general", "kenobi"], ["hello", "there", "!"]],  # tokenized references for the first sample (2 references)
#     [["foo", "bar", "foobar"]]                                           # tokenized references for the second sample (1 reference)
# ]

references = [
    [["hello", "there", "general", "kenobi"]],  # tokenized references for the first sample (2 references)
    [["foo", "bar", "foobar", "conf"]]                                           # tokenized references for the second sample (1 reference)
]

metric_name = 'bleu'
metric = load_metric(metric_name)
result = metric.compute(predictions=predictions, references=references)
print(result)
result = {"bleu": result["bleu"]}
print(result)



# print ("###########")
# predictions = ["hello there general kenobi"]
# references = ["hello there general kenobi"]
# bleu_score = compute_metrics(predictions, references, metric_name='bleu')
# print(bleu_score)