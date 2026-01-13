# Two Tower Candidate Retrieval

The following implementation is based on the insights gathered from two key papers: the original YouTube paper introducing the two-tower architecture ([1](https://dl.acm.org/doi/pdf/10.1145/2959100.2959190)), and a later paper from Google detailing its use at scale ([2](https://storage.googleapis.com/gweb-research2023-media/pubtools/5716.pdf)).

### Dataset

A suitable dataset for this task requires user-item interactions with both implicit and explicit feedback, as well as rich features for users, items, and context. The [Yandex/Yambda dataset](https://arxiv.org/abs/2505.22238) meets these requirements at a massive scale. This project will use the `flat-multievent-500m` version, which contains 500 million interactions.

### Objective

The goal is to design an efficient candidate retrieval system with strong recall performance. For the Yambda dataset, this means retrieving relevant music items given a user's context and a seed track. The problem is framed as predicting the **"next" track** a user will listen to, based on a query composed of user features, the current seed track, and historical listening patterns.

### Evaluation Metric

**Recall@k**: This metric measures how effective the model is at retrieving the true "next" items that a user listens to within top `k` predictions. For a given `<user, seed_track>` pair, the model generates a list of recommended tracks. `Recall@k` is the percentage of all test cases where the true next item was successfully retrieved within the top `k` recommendations. A higher `Recall@k` indicates that the model is better at not missing relevant items.

### Positive and Negative Samples

-   **Positive Samples**: These are defined by implicit feedback, specifically users listening to tracks. This provides a much larger and more natural dataset than explicit feedback (likes/dislikes). The loss function will incorporate weights to reflect that not all positive interactions (e.g., listening for 10% vs. 90% of a track) are equal.
-   **Negative Samples**: All unobserved user-item pairs are potential negative samples. This project will use in-batch negative sampling, a technique described in the 2019 paper [2], which uses other items in a training batch as negatives. This method is efficient but introduces a sampling bias that must be corrected in the loss function.

### Features

-   **User Features**
    -   `user_id`
-   **Contextual (Interaction) Features**
    -   `play_ratio` (implicit feedback)
    -   `event_type` (listen, like, dislike, etc.)
-   **Item Features**
    -   `item_id`
    -   `track_length`
    -   CNN audio spectrogram embedding
    -   `artist_id`
    -   `album_id`

### Model

Given a query (user + context), retrieve relevant candidate items.

The Two-Tower Model consists of:
-   **Query Tower**: Encodes the user's context.
    -   **User History Features**:
        -   Embedding of the last N listened-to items (e.g., average embedding).
        -   Embedding of the last N listened-to artists (e.g., average embedding).
        -   Embedding of the last N listened-to albums (e.g., average embedding).
        -   `history_track_length` (average across N items).
        -   pre-computed audio embedding (average across N items).
    -   **Seed Track Features**:
        -   `item_id` embedding.
        -   `artist_id` embedding.
        -   `album_id` embedding.
        -   pre-computed audio embedding.
        -   `track_length`
    -   **User Identity Features**:
        -   `user_id` embedding.
        -   age, gender, demographics, device... (not available)
    -   **Other**
        -   Example Age/age of request (Train (time since last interaction, not upload time): Train_max_timestamp - event_timestamp; Test: 0 or -1.e-9 "what is most likely to be listened to right now")
            - Example age captures time dependent trendiness of an item: video A might have had a lot more overall interactions than video B, but interactions for video A were primarily 30 days ago whereas interactions for video B were 1-2 days ago. During training, model learns that when age is close to 0, probability of click is high for new/trending videos. When age is high, probability of click is high for older videos. In this case, during inference, when age = 0, model will give higher probability to video B because it was trending more recently. without this feature, model would not capture time dependent trends and would predict average probability over time, which is higher for older videos since they have more time to accumulate interactions.
-   **Item (Candidate) Tower**: Encodes a potential candidate item.
    -   **Item Features**:
        -   `item_id` embedding.
        -   `artist_id` embedding.
        -   `album_id` embedding.
        -   `track_length`
        -   pre-computed audio embedding.
        -   track release date (not available)

### Retrieval

Given a user request, we use query tower to generate a query embedding, which is used to look up the top k item embeddings. In order to make this process faster, item embeddings should be stored in a data structure that allows fast retrieval. The most accurate algorithm would be to use knn, but brute‑force search over millions of items is very expensive (O(NxD) time and O(NxDx4) memory). There are many tools that approximate knn (ANN) using various techniques. They often trade off between speed, memory, and accuracy: ANNOY (Spotify), Scann(Google), Faiss(Meta). I chose Faiss because it offers many techniques to experiment with. Specifically, I use IVF to cluster item embeddings (speed) + PQ to make it memory efficient.

### Concepts covered
- In-Batch Negative Sampling
- Bias correction
- Frequency Estimation using hashes and weighted moving average
- Index Table techniques: IVF, PQ, HNSW


### Links
- [Deep Neural Networks for YouTube Recommendations (2016)](https://dl.acm.org/doi/pdf/10.1145/2959100.2959190)
- [Sampling-Bias-Corrected Neural Modeling for Large Corpus
Item Recommendations (2019)](https://storage.googleapis.com/gweb-research2023-media/pubtools/5716.pdf)
- [Yandex/Yambda dataset](https://arxiv.org/abs/2505.22238)
- [Faiss Documentation](https://faiss.ai/index.html)
- [IVF](https://medium.com/data-science/similarity-search-blending-inverted-file-index-and-product-quantization-a8e508c765fa)
- [PQ](https://medium.com/data-science/similarity-search-product-quantization-b2a1a6397701)
- [HNSW+IVF+PQ](https://medium.com/data-science/similarity-search-part-4-hierarchical-navigable-small-world-hnsw-2aad4fe87d37)


