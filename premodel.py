import json
import pickle
import pprint
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from preprocess import preprocess_text
from topicModel import get_topics

import matplotlib.pyplot as plt


def recommendation(query, tfidf_model, tfidf_matrix, paperTitles):
    """
        Takes the query, performs basic preprocessing. Converts it into a
        tfidf vector. Then returns a list of paper titles most similar to
        the query based on cosine similarity.

        Parameters
        ----------
        query : string
            Search query for the paper.
        tfidf_model : sklearn.feature_extraction.text.TfidfVectorizer
            TfidfVectorizer model
        tfidf_matrix : np.array
            Precomputed TF-IDF vectors of paper summaries.
        paperTitles : list of string
            List of titles of papers
     """
    # Preprocess the query
    processedQuery = np.array(list(map(preprocess_text, query)))

    tfidfQuery = tfidf_model.transform(processedQuery)

    cosine_similarities = linear_kernel(tfidf_matrix, tfidfQuery)
    related_docs_indices = np.argsort(
        cosine_similarities, axis=0)[-10:].reshape((-1, ))[::-1]

    result = []
    for id in related_docs_indices:
        result.append(paperTitles[id])

    return result


def get_related_topic(query, nmf_model, tfidf_model, topic_dict, num_top_words=10):

    processedQuery = np.array(list(map(preprocess_text, query)))
    vectQuery = tfidf_model.transform(processedQuery)

    # Gives the probability of each topic for the query
    # in a matrix of (top_words * no_of_topics) form
    # i.e. which topic does each word belong to.
    topic_probability_scores = nmf_model.transform(vectQuery)
    query_topic = np.argmax(np.sum(topic_probability_scores, axis=0))
    return query_topic, topic_dict[query_topic]


def get_papers_per_year(data_dir, data, topic_no):
    with open(data_dir + "topic_labels.pk", "rb") as fp:
        topic_labels = pickle.load(fp)

    idxs = np.where(topic_labels == topic_no)[0]

    years = [data[i]["year"] for i in idxs]

    return Counter(years)


if __name__ == "__main__":
    data_dir = "./data/"

    with open(data_dir + "arxivData.json", "r") as fp:
        data = json.load(fp)

    paperTitles = list(map(lambda x: x["title"], data))

    with open(data_dir + 'vectorizer.pk', 'rb') as pickle_in:
        vectorizer = pickle.load(pickle_in)

    with open(data_dir + "tfidf-vectors-200.pk", "rb") as fp:
        vectSum = pickle.load(fp)

    with open(data_dir + "nmf_model.pk", 'rb') as fp:
        nmf_model = pickle.load(fp)

    with open(data_dir + "topic_dict.pk", 'rb') as fp:
        topic_dict = pickle.load(fp)

    query = np.array(
        ["Image processing"], dtype=object)

    recommendedList = recommendation(query, vectorizer, vectSum, paperTitles)
    topic_no, topic_name = get_related_topic(
        query, nmf_model, vectorizer, topic_dict)

    per_year_count = get_papers_per_year(data_dir, data, topic_no)

    pprint.pprint(recommendedList)

    years = list(per_year_count.keys())
    years = sorted(years)
    counts = [per_year_count[i] for i in years]
    years = list(map(str, years))

    plt.barh(years, counts)
    plt.title("Papers published related to the topic per year")
    plt.show()
