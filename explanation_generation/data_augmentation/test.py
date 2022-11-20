from fitbert import FitBert


# in theory you can pass a model_name and tokenizer, but currently only
# bert-large-uncased and BertTokenizer are available
# this takes a while and loads a whole big BERT into memory
model_name="distilbert-base-uncased"
model_name="bert-large-uncased"
fb = FitBert(model_name=model_name)

masked_string = "Why Bert, you're looking ***mask*** today!"
options = ['buff', 'handsome', 'strong']

masked_string = "a clock can tell you time, so clock and time have a  ***mask*** relation"
options = ['function', 'part of', 'location']

ranked_options = fb.rank(masked_string, options=options)
print(ranked_options)







