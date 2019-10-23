# Libraries required to be imported
import json
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def str_to_list(x):
    """
        Converting string enclosed lists to a python list
    """
    for i in range(len(x)):
        x[i]["author"] = eval(x[i]["author"])
        # x[i]["link"] = eval(x[i]["link"])
        x[i]["tag"] = eval(x[i]["tag"])


def preprocess_text(text):
    """
        Preprocess text data for performing analysis
    """
    # Converting all the letters to lowercase
    text = text.lower()
    # Removing numbers from the text
    text = re.sub(r"\d+", "", text)
    # Removing punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Removing trailing and leading whitespaces
    text = text.strip(" ")
    # Removing stop words
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filt_sum = [w for w in word_tokens if not w in stop_words]

    filt_sum = " ".join(filt_sum)

    return filt_sum


# Directory where data is stored
DATA_DIR = "data/"

with open(DATA_DIR + "arxivData.json", "r") as fp:
    data = json.load(fp)

str_to_list(data)

# Get each paper's abstract/summary
paperSummaries = np.array(list(map(lambda x: x["summary"], data)))

cleanSummaries = np.array(list(map(preprocess_text, paperSummaries)))
# TODO: Stem the words later if needed

# Converting each abstract into a TF-IDF vector
vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
vectSum = vectorizer.fit_transform(cleanSummaries)

np.save(DATA_DIR + "tfidf-vectors.npy", vectSum)
