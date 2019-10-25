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
    related_docs_indices = cosine_similarities[0].argsort()[:-5:-1]
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
    
    
    query = np.array(["Recent approaches Multiresolution Recurrent Neural Networks: An Application to Dialogue\n  Response Generation based on artificial neural networks (ANNs) have shown\npromising results for short-text classification.We report on the production experience of the D0 experiment at the Fermilab\nTevatron, using the SAM data handling system with a variety of computing\nhardware configurations, batch systems, and mass storage strategies. We have\nstored more than 300 TB of data in the Fermilab Enstore mass storage system. We\ndeliver data through this system at an average rate of more than 2 TB/day to\nanalysis programs, with a substantial multiplication factor in the consumed\ndata through intelligent cache management. We handle more than 1.7 Million\nfiles in this system and provide data delivery to user jobs at Fermilab on four\ntypes of systems: a reconstruction farm, a large SMP system, a Linux batch\ncluster, and a Linux desktop cluster. In addition, we import simulation data\ngenerated at 6 sites worldwide, and deliver data to jobs at many more sites. We\ndescribe the scope of the data handling deployment worldwide, the operational\nexperience with this system, and the feedback of that experience. A knowledge base is redundant if it contains parts that can be inferred from\nthe rest of it. We study the problem of checking whether a CNF formula (a set\nof clauses) is redundant, that is, it contains clauses that can be derived from\nthe other ones. Any CNF formula can be made irredundant by deleting some of its\nclauses: what results is an irredundant equivalent subset (I.E.S.) We study the\ncomplexity of some related problems: verification, checking existence of a\nI.E.S. with a given size, checking necessary and possible presence of clauses\nin I.E.S.'s, and uniqueness. We also consider the problem of redundancy with\ndifferent definitions of equivalence.However, many short texts\noccur in sequences (e.g., sentences in a document or utterances in a dialog),\nand most existing ANN-based systems do not leverage the preceding short texts\nwhen classifying a subsequent one. In this work, we present a model based on\nrecurrent neural networks and convolutional neural networks that incorporates\nthe preceding short texts. Our model achieves state-of-the-art results on three\ndifferent datasets for dialog act prediction."], dtype=object)
    recommendedList = recommendation(query, vectorizer, vectSum, idSummaryMap)
    

    print(recommendedList)
