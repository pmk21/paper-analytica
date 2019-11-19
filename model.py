import json
import pickle
import pprint
from collections import Counter

import fire
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from preprocess import preprocess_text
from topicModel import get_topics


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

        Returns
        -------
        result : list
            Top 10 titles of research papers
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


def plot_count_dict(cdict, title, sort='key'):
    if sort == 'key':
        items = cdict.items()
        items = sorted(items, key=lambda x: x[0])
    elif sort == 'value':
        items = cdict.items()
        items = sorted(items, key=lambda x: x[1])
    else:
        raise ValueError("sort takes either 'key' or 'value'")

    labels = [i[0] for i in items]
    counts = [i[1] for i in items]
    plt.barh(labels, counts)
    plt.title(title)
    plt.show()


def get_papers_per_year(data_dir, data, topic_no):
    with open(data_dir + "topic_labels.pk", "rb") as fp:
        topic_labels = pickle.load(fp)

    idxs = np.where(topic_labels == topic_no)[0]

    years = [data[i]["year"] for i in idxs]

    return Counter(years)


def top_authors(data_dir, data, topic_no):
    with open(data_dir + "topic_labels.pk", "rb") as fp:
        topic_labels = pickle.load(fp)

    idxs = np.where(topic_labels == topic_no)[0]

    authors = []

    for i in idxs:
        temp_auth = eval(data[i]["author"])
        for j in range(len(temp_auth)):
            authors.append(temp_auth[j]["name"])

    return Counter(authors)


def model(search_query, data_dir="./data/"):
    """
        Takes in a query and path to the data and returns the list of
        top 10 papers related to the query as well as 2 horizontal bar
        plots

        Parameters
        ----------
        search_query : string
            Query to be searched for in the dataset.
        data_dir : string
            Path to the directory storing the data required.
    """
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
        [search_query], dtype=object)

    recommendedList = recommendation(query, vectorizer, vectSum, paperTitles)
    topic_no, _ = get_related_topic(
        query, nmf_model, vectorizer, topic_dict)

    pprint.pprint(recommendedList)

    per_year_count = get_papers_per_year(data_dir, data, topic_no)
    plot_count_dict(
        per_year_count, "Papers published related to the topic per year")

    auth_count = top_authors(data_dir, data, topic_no)
    plot_count_dict(dict(auth_count.most_common(10)),
                    "Top authors and number of papers",
                    "value")


if __name__ == "__main__":
    fire.Fire(model)