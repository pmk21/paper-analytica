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
from preprocess import preprocess_text, str_to_list


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


if __name__ == "__main__":
    data_dir = "./data/"
    
    with open(data_dir + "arxivData.json", "r") as fp:
            data = json.load(fp)
    str_to_list(data)

    paperSummaries = np.array(list(map(lambda x: x["summary"], data)))
    idSummaryMap = list(map(lambda x:(x["id"], x["summary"]), data))
    cleanSummaries = np.array(list(map(preprocess_text, paperSummaries)))

    pickle_in = open(data_dir + 'vectorizer.pk', 'rb')
    vectorizer = pickle.load(pickle_in)
    # vectSum = np.load('./data/tfidf-vectors-100.npy', allow_pickle = True)
    vectSum = vectorizer.fit_transform(cleanSummaries)
    
    query = np.array(["Recent approaches based on artificial neural networks (ANNs) have shown\npromising results for short-text classification. However, many short texts\noccur in sequences (e.g., sentences in a document or utterances in a dialog),\nand most existing ANN-based systems do not leverage the preceding short texts\nwhen classifying a subsequent one. In this work, we present a model based on\nrecurrent neural networks and convolutional neural networks that incorporates\nthe preceding short texts. Our model achieves state-of-the-art results on three\ndifferent datasets for dialog act prediction."], dtype=object)
    recommendedList = recommendation(query, vectorizer, vectSum, idSummaryMap)
    

    print(recommendedList)
