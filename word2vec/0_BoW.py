import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import sys

BoW_COUNT = int(sys.argv[1])

train = pd.read_csv("./data/labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

def review_to_words(review):
    review_text = BeautifulSoup(review).get_text()
    letters_only = re.sub("[^a-zA-Z]"," ", review_text)

    words = letters_only.lower().split()

    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops] 

    return (" ".join(meaningful_words))

clean_review = review_to_words(train["review"][0])

num_reviews = train["review"].size
train_reviews = []

for i in xrange(0, num_reviews):
    if((i+1)%1000 == 0):
        print "Review %d of %d" % (i+1, num_reviews)
    train_reviews.append(review_to_words(train["review"][i]))

print "Creating the bag of words"
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = BoW_COUNT)

train_data_features = vectorizer.fit_transform(train_reviews)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()
dist = np.sum(train_data_features, axis=0)

print " Top 10"
for idx, word in enumerate(vocab[:10]):
    print idx, word

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, train['sentiment'])

test = pd.read_csv("./data/testData.tsv", header=0, delimiter='\t', quoting=3)

num_reviews = len(test['review'])
clean_test_reviews = []

for i in xrange(0, num_reviews):
    if ((i+1) % 1000 == 0):
        print "Reviews %d of %d" % (i+1, num_reviews)
    clean_review = review_to_words(test['review'][i])
    clean_test_reviews.append(clean_review)

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

output = pd.DataFrame(data = {"id": test['id'], 'sentiment': result})


output.to_csv("BoW_%s.csv" % BoW_COUNT, index=False, quoting=3)
