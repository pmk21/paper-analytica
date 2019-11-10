# Libraries required to be imported
import json
import os
import pickle
import re
import string
import sys

import fire
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def str_to_list(x):
    """
        Converting string enclosed lists to a python list.
    """
    for i in range(len(x)):
        x[i]["author"] = eval(x[i]["author"])
        # x[i]["link"] = eval(x[i]["link"])
        x[i]["tag"] = eval(x[i]["tag"])

escapes = ''.join([chr(char) for char in range(1, 32)])
esc_dict = {ord(c): " " for c in escapes}

def preprocess_text(text, stemmer=PorterStemmer(), esc_dict=esc_dict):
    """
        Preprocess text data for performing analysis.
    """
    # Converting all the letters to lowercase
    text = text.lower()
    # Removing numbers from the text
    text = re.sub(r"\d+", "", text)
    # Removing punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove escape characters
    text = text.translate(esc_dict)
    # Removing trailing and leading whitespaces
    text = text.strip(" ")
    # Removing stop words
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filt_sum = [w for w in word_tokens if w not in stop_words]

    # Stemming the words
    filt_sum = [stemmer.stem(w) for w in filt_sum]

    filt_sum = " ".join(filt_sum)

    return filt_sum


def compute(data_dir="./data/", num_features=100):
    """
        Computes the TF-IDF vector and saves it.
        The TF-IDF model is also saved.

        Parameters
        ----------
        data_dir : string
            Path to the directory containing the data.
        num_features : int
            Max words to consider in computing TF-IDF values.
    """

    if data_dir == "./data/" and not os.path.isdir(data_dir):
        os.mkdir("./data")

    try:
        with open(data_dir + "arxivData.json", "r") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        print('Data does not exist in "{0}" !'.format(data_dir))
        sys.exit(1)

    str_to_list(data)

    # Get each paper's abstract/summary
    paperSummaries = np.array(list(map(lambda x: x["summary"], data)))

    cleanSummaries = np.array(list(map(preprocess_text, paperSummaries)))

    # Converting each abstract into a TF-IDF vector
    vectorizer = TfidfVectorizer(
        tokenizer=word_tokenize, max_features=num_features)
    vectSum = vectorizer.fit_transform(cleanSummaries)

    with open(data_dir + "tfidf-vectors-" + str(num_features) + ".pk", 'wb') as fp:
        pickle.dump(vectSum, fp)

    print("Computed vector and saved!")

    with open(data_dir + 'vectorizer.pk', 'wb') as fin:
        pickle.dump(vectorizer, fin)

    print("Saved TF-IDF vectorizer!")


if __name__ == "__main__":
    fire.Fire(compute)
