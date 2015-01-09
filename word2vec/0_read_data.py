import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

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
print clean_review

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
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(train_reviews)
train_data_features = train_data_features.toarray()

print "BoW : %s" % train_data_features.shape
