# Content-Based Research Paper Recommendation and Analytics Engine

## About

At this time and age, research progresses exponentially and a lot of research papers get published everyday making it hard for a user to find a genuinely good research paper which is relevant to his/her field of research. We plan to solve this problem by analyzing research papers and providing the best papers relevant to the query of the user. We also provide relevant analytics related to the query as well as to each paper.

## Dataset

We plan to use [ARXIV data from 31000+ papers](https://www.kaggle.com/neelshah18/arxivdataset) which is present on Kaggle. This data is mostly restricted to computer science. It contains metadata of all papers related to machine learning, computational language, neural and evolutionary computing, artificial intelligence, and computer vision fields published between 1992 to 2018.

## Usage

In order to obtain the results mentioned in the report follow the below steps -

1. First clone the repo to your local machine.

2. Download the dataset mentioned above and place it in the data directory which is present in the same directory where you will be running the program. This is how it should look like -

    ```shell
    .
    ├── data
    │   └── arxivData.json
    ├── EDA.ipynb
    ├── LICENSE
    ├── model.py
    ├── preprocess.py
    ├── README.md
    └── topicModel.py
    ```

3. After completing the above step, run `preprocess.py`

    ```shell
    $ python preprocess.py
    Computed vector and saved!
    Saved TF-IDF vectorizer!
    ```

    You can also use `python preprocess.py --help` for additional options.

4. After running `preprocess.py`, run `topicModel.py`

    ```shell
    $ python topicModel.py
    NMF model saved!
    Saved topic dictionary!
    Saved topic labels!
    ```

5. After the topics have been computed, run `model.py`

    ```shell
    $ python model.py "clustering techniques"
    ['An Analysis of Gene Expression Data using Penalized Fuzzy C-Means\n'
     '  Approach',
     'A Comparative study Between Fuzzy Clustering Algorithm and Hard\n'
     '  Clustering Algorithm',
     'On comparing clusterings: an element-centric framework unifies overlaps\n'
     '  and hierarchy',
     'Sparse Convex Clustering',
     'Similarity-Driven Cluster Merging Method for Unsupervised Fuzzy\n'
     '  Clustering',
     'Functorial Hierarchical Clustering with Overlaps',
     'Adaptive Evolutionary Clustering',
     'An Analytical Study on Behavior of Clusters Using K Means, EM and K*\n'
     '  Means Algorithm',
     'Clustering Multidimensional Data with PSO based Algorithm',
     'Risk Bounds For Mode Clustering']
    ```

    The above command will also generate 2 graphs.

    You can use `$ python model.py --help` for additional options.
