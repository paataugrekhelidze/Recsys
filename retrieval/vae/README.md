# Variational Autoencoders for Collaborative Filtering

This project is an adaptation of the [Netflix paper](https://arxiv.org/abs/1802.05814), which explores the transition from linear factor models to nonlinear models (VAEs) for Netflix movie ranking. The paper demonstrates significant improvements in metrics such as Recall@R and NDCG@R when comparing the proposed nonlinear model to linear models like [SLIM](https://www.researchgate.net/publication/220765374_SLIM_Sparse_Linear_Methods_for_Top-N_Recommender_Systems) and [WMF](http://yifanhu.net/PUB/cf.pdf).

## Goals
- Explain the methodology
- Recreate the VAE model architecture
- Improve upon the metrics recorded in the Netflix paper
- Stress test the model at production-level volume