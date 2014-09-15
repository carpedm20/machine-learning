#!/usr/bin/python
# NaiveBayesClassifier for movie review data set from nltk
import nltk
import random
from nltk import NaiveBayesClassifier

movie_reviews = nltk.corpus.movie_reviews
cat = movie_reviews.categories()

ids = {}
for c in cat:
    ids[c] = movie_reviews.fileids(c)
    random.shuffle(ids[c])

train_files = {}
test_files = {}

for id in ids:
    current_files = ids[id]

    test_files[id] = current_files[:len(current_files)/4]
    train_files[id] = current_files[len(current_files)/4:]

def feature_extraction(file):
    features = []
    for id in ids:
        current_files = file[id]
        
        for f in current_files:
            words = movie_reviews.words(fileids=[f])

            features.append((dict((word, True) for word in words), id))
            
    return features

train_set = feature_extraction(train_files)
test_set = feature_extraction(test_files)

classifier = nltk.NaiveBayesClassifier.train(train_set)
print ("accuracy : %s" % nltk.classify.accuracy(classifier, test_set))

# classifier.classify({'good':True,'bad':True,'excellent':True})
