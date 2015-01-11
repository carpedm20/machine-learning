import pandas as pd

from utils import *

labeled_train = "./data/labeledTrainData.tsv"
unlabeled_train = "./data/unlabeledTrainData.tsv"
test = "./data/testData.tsv"

train = pd.read_csv(labeled_train, header=0, delimiter="\t", quoting=3)
test = pd.read_csv(test, header=0, delimiter="\t", quoting=3)

unlabeled_train = pd.read_csv(unlabeled_train, header=0, 
                               delimiter="\t", quoting=3)


import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)


import logging
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300 # Word vector dimensionality                      
min_word_count = 40
num_workers = 12
context = 10 # Context window size                                                                                    
downsampling = 1e-3 # Downsample setting for frequent words

print " [*] Word2vec Start... "
model = word2vec.Word2Vec(sentences, workers=num_workers, \
                          size=num_features, min_count = min_word_count, \
                          window = context, sample = downsampling)

model.init_sims(replace=True)

model_name = "300features_40minwords_10context"
model.save(model_name)
