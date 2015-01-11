from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_words(review):
    review_text = BeautifulSoup(review).get_text()
    letters_only = re.sub("[^a-zA-Z]"," ", review_text)

    words = letters_only.lower().split()

    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops] 

    return (" ".join(meaningful_words))

def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences
