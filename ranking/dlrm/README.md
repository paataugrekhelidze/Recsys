### Deep Learning Recommendation

#### Architecture
The neighborhood based models, such as two towers + index table for ANN, have a weakness. The towers train the item and user towers separately and they only interact through a dot product. This misses complex nonlinear interactions between users and items. Preranking helps identify such patterns with more precision than two-tower model, and it is small enough to process ~1000 queries within acceptable latency. However, ranking model includes more features and bigger model. It will increase precision even more, but we make sure that it receives small enough candidates (to meet SLA latency) with good enough precision via preranking.
- Preranking - Distilled/Smaller DLRM or Logistic Regression (~1000 to ~200)
- Ranking - Memory-intensive (RMC2) or Compute-intensive (RMC3) DLRM (~200 to 10)

#### Dataset
In this project I wanted focus more on scale, so the project uses the Criteo 1TB Ad Click dataset. However, I wanted to be consistent with paper [1] so I will be evaluating the model against the smaller, Kaggle version of of the Criteo Dataset. It consists of 13 integer and 26 categorical features, where the first column represents binary values of ad click (target). Goal will be to maximize the CTR, is this an ideal objective? probably not, since the model might just learn to recommend clickbait.


#### Offline Eval
- Normalized Entropy / Normalized Entropy rank
- Calibration / Calibration rank
- AUC / AUC rank

#### Obejctives
- replicate the performance of the original DLRM paper [1] on the Cretioe Kaggle Dataset [4]
- Move to a larger, Criteo 1TB dataset[2], which will at least require DDP (exploring ray train)
- Deploy and serve at least under 100ms with stable range between p5 and p99

#### Links:
1. [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091)
2. [The Architectural Implications of Facebook’s DNN-based Personalized Recommendation](https://arxiv.org/pdf/1906.03109)
3. [DLRM Github Repository](https://github.com/facebookresearch/dlrm/blob/main/dlrm_s_pytorch.py)
4. [Criteo 1TB Click Logs Dataset](https://huggingface.co/datasets/criteo/CriteoClickLogs/blob/main/README.md)
5. [Criteo Kaggle Dataset](https://www.kaggle.com/competitions/criteo-display-ad-challenge/overview)