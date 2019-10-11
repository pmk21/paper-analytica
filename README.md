# Content-Based Research Paper Recommendation and Analytics Engine

## About

At this time and age, research progresses exponentially and a lot of research papers get published everyday making it hard for a user to find a genuinely good research paper which is relevant to his/her field of research. We plan to solve this problem by analyzing research papers and providing the best papers relevant to the query of the user. We also provide relevant analytics related to the query as well as to each paper.

The main purpose of this Recommendation and Analytics Engine is not solely to recommend the research paper but also to provide relevant analytics such that the recommendation is influenced in a positive way.The recommendation engine is built using content based filtering algorithm which best suits our application where, as of now, the users are not distinguished.The User inputs a query which is preprocessed by removing stopwords if any and is converted into a TFIDF vector. This vector is compared with the existing vectors corresponding to the research papers in the database and the one with the best cosine similarity, basically the projection of one vector over another, is chosen and the resesarch corresponding to the same is given as a output to user. Along with this, the paper's topics is modelled and which is used to tag the paper.   
