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
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def str_to_list(x):
    """
        Converting string enclosed lists to a python list.
    """
    for i in range(len(x)):
        x[i]["author"] = eval(x[i]["author"])
        # x[i]["link"] = eval(x[i]["link"])
        x[i]["tag"] = eval(x[i]["tag"])


def preprocess_text(text):
    """
        Preprocess text data for performing analysis.
    """
    # print(type(text))
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
    filt_sum = [w for w in word_tokens if w not in stop_words]

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
    idSummaryMap = list(map(lambda x:(x["id"], x["summary"]), data))
    cleanSummaries = np.array(list(map(preprocess_text, paperSummaries)))
    # TODO: Stem the words later if needed

    # Converting each abstract into a TF-IDF vector
    vectorizer = TfidfVectorizer(
        tokenizer=word_tokenize, max_features=num_features)
    vectSum = vectorizer.fit_transform(cleanSummaries)
    print(len(vectSum.toarray()))

    np.save(data_dir + "tfidf-vectors-" + str(num_features) + ".npy", vectSum)

    print("Computed vector and saved!")
    query = ["Recent approaches based on artificial neural networks (ANNs) have shown\npromising results for short-text classification. However, many short texts\noccur in sequences (e.g., sentences in a document or utterances in a dialog),\nand most existing ANN-based systems do not leverage the preceding short texts\nwhen classifying a subsequent one. In this work, we present a model based on\nrecurrent neural networks and convolutional neural networks that incorporates\nthe preceding short texts. Our model achieves state-of-the-art results on three\ndifferent datasets for dialog act prediction."]
    query = np.array(query)
    recommendedList = recommendation(query, vectorizer, vectSum, idSummaryMap)


def recommendation(query, tf, tfidf_matrix, idSummariesMap):
    """ 
        Takes the query, performs basic nlp processing, 
        fits it into the tfidf vector space model, 
        returns a list of the most similar papers using the cosine similarity on the model.
     """
    # process the query 
    processedQuery = np.array(list(map(preprocess_text, query)))
    
    tfidfQuery = tf.fit_transform(processedQuery)
    cosine_similarities = linear_kernel(tfidfQuery, tfidf_matrix)
    related_docs_indices = cosine_similarities[0].argsort()[:-10:-1]
    result = []
    for idx, row in enumerate(idSummariesMap):
        # print(idx)
        if(idx in related_docs_indices):
            result.append(row[0])
    
    return result





    with open(data_dir + 'vectorizer.pk', 'wb') as fin:
        pickle.dump(vectorizer, fin)

    print("Saved TF-IDF vectorizer!")


if __name__ == "__main__":
    fire.Fire(compute)
