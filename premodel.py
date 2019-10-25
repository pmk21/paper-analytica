import json
import pickle
import pprint

import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from preprocess import preprocess_text


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


if __name__ == "__main__":
    data_dir = "./data/"

    with open(data_dir + "arxivData.json", "r") as fp:
        data = json.load(fp)

    paperTitles = list(map(lambda x: x["title"], data))

    with open(data_dir + 'vectorizer.pk', 'rb') as pickle_in:
        vectorizer = pickle.load(pickle_in)

    with open(data_dir + "tfidf-vectors-200.pk", "rb") as fp:
        vectSum = pickle.load(fp)

    query = np.array(["Manifold analysis and dimensionality reduction"], dtype=object)
    recommendedList = recommendation(query, vectorizer, vectSum, paperTitles)

    pprint.pprint(recommendedList)
