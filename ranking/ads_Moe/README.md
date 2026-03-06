### Multi-Task Learning using Multi-Gate MoE

Recommendation tasks often involve multiple objectives that represent different aspects of user utility. Instead of creating separate models for each task, it is computationally more efficient to build a single model with multiple task heads. The shared-bottom layer has been used to share low-level feature representations across tasks. However, its performance decreases when tasks are unrelated, as the model struggles to learn conflicting representations within shared layers. MoE modularizes the architecture, allowing conflicting patterns to be learned by different experts. By using a Multi-Gate variation of MoE (one gate per task), independent gates can capture both shared task information (shared experts) and task-specific information (specialized experts and task heads).

The YouTube paper [3] combines concepts from the MMoE paper [2] and the Wide & Deep paper [5], representing the deep part using MMoE and introducing an auxiliary shallow network to learn selection bias. Why a shallow network? Because we hand-engineer feature crosses (explicit feature interaction) that are assumed to produce the selection bias, specifically, the combination of the item's position in the UI and the device used. These crosses are memorized in the shallow network, and its output is combined with the main model's output. The goal is to force the shallow network to learn the bias, allowing the main model to focus on relevance. At inference, we perform ablation by setting the position value to "missing," leveling the ground between samples so that the competition primarily comes from the deep/main model (MMoE), which focuses on sample relevance.

The Meta paper [6] focuses more on the early / pre-ranking stage and discusses challenges such as: 
- misalignment of the total ad score between early and final ranking stages, ads that would've scored high in the final stage are lost due to them getting low scores in the early stage
- selection bias, training data for early stage model is biased towards the data it trains on (impression data), introducing a feedback loop. How to provide signals from non-impression data, which consists of the majority of data?

The Meta paper [7] is old (2014), but it discusses a simple model, online learning for a model, and how to generate data using an online joiner that combines impression and action streams. Can the same online learning technique be used on models from paper [3] and [6]? Does the update happen through a central model artifact and then to the serving instances? to be determined...

**Objectives:**
- Define a final ranker, a multi-task problem, similar to paper [3], but adjusted for DLRMv2 [11]
    - Identify features contributing to selection bias (position bias)
    - Build a shallow network to learn position bias
    - Build a deep network (Options: MLP-only, DCNv2, DLRMv1, DLRMv2) -> focus on DLRMv2
    - Use dropout in the gate network and importance loss to prevent gating network polarization
    - Use feature dropout on shallow network to prevent over-reliance on position features
    - Display expert utilization (ideally, all experts should be utilized, with more specialization for unrelated tasks) -> compare MMoE vs PLE
- Define an early-stage model, similar to paper [6]
    - two-tower DLRM for feature transformation, historic user and ad embedding can be cached for faster inference.
    - perform a task distillation from a teacher model (improve consistency with final stage model, which improves overall system performance, even at the cost of a per task performance)
    - shared-bottom layer with multiple task heads for CTR, CVR, CTR distillation, and CVR distillation
- Define an alternative early-stage model, multiple models for each task, similar to paper [7]
    - boosted decision trees for feature transformation
    - logistic regression for ranking (fast, simple online learning)

**Evaluation:**

Online metrics (system specific):
- Cross-out rate (if we had one, we don't). Paper [6] section 4
- Ads survey for quality, ASQ (we don't have one). Paper [6] section 4
- Impression-based total value, where total value is a combination of actions, ad bid, and ad quality. Paper [6] section 4
- Click-through rate, CTR
- Conversion Rate, CVR
- Total value divergence (for early stage model). Paper [6] section 4

Offline metrics (task specific):
- CVR and CTR are both binary classification, both Normalized Entropy (NE) and AUC metrics could work to measure per task performance. Paper [7] section 2
    - NE and AUC do not quite capture the online performance, we need something that is closer to the online objective (like total value (soft recall), which captures quality, relevance, and bid price).
- soft recall: given N candidates, and top-K total value scores from the final stage (golden set), soft recall = sum of total value of early stage model top-k ads / sum of total value of the golden set. Paper [6] section 3.3

**Dealing with non-stationary Embeddings at Scale**
Ads/users are constantly being created and deleted, creating and maintaining functional embedding vectors at scale becomes infeasible. Solution, use hashing technique for user and ad embeddings. This allows us to define a trade-off between memory and accuracy (collision rate). Collisions can be dealt with double hashing, which increases computation latency (use frequency based double hashing to prevent heavy computation on most frequent features, paper [12] Figure 3). hashing will also help with cold-start problem, where new items are assigned to vector embeddings with pre-trained semantics. Frequency based hashing requires us to use two independent hashing algorithms + promote and pop indices from top-k most frequent list. This can get complicated and slow down this project, currently using quotient-remainder hashing, which increases the representation space quadratically while keeping the same memory.


**Calibration**
For ads ranking, it is important that the probabilities are well calibrated, meaning that the model scores can be interpreted as long-run frequencies (ex: samples that predict 0.8 should have a a ground-truth fraction of 80% for the event). This allows for a better estimation for the actual cost so the advertisers do not overpay/underpay for the ad.
- Measure calibration visually with the calibration curve. Creates bins (adjustable), for samples within a bin, draws average of predicted probs vs fraction of ground truth (ideal is a diagonal).
- Brier Score - calibration is not everything, predicting class average would give us a perfect calibration model bad model. Brier score takes into account a balance beteween calibration and sharpness, where sharpness measure the deviation from the global average. If model is overconfident, it predicts tails (predicted: 99%, actual: 80%, predicted: 1%, actual: 20%) we lose sharpness to improve calibration. By avoiding overcondifent/underconfident scores, we can improve both calibration and sharpness.
- I-spline - learn a model on top of a black-box ranker which takes current scores and returns calibrated probabilites. Compare calibrated model vs uncalibrated via calibration curve and brier score. AUC should stay similar since calibration is posible without affecting AUC.

**Links**
1. [OUTRAGEOUSLY LARGE NEURAL NETWORKS:
THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER](https://arxiv.org/pdf/1701.06538)
2. [Modeling Task Relationships in Multi-task Learning with
Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/epdf/10.1145/3219819.3220007)
3. [Recommending What Video to Watch Next: A Multitask
Ranking System](https://daiwk.github.io/assets/youtube-multitask.pdf)
4. [Mixture-of-Experts based Recommender Systems](https://blog.reachsumit.com/posts/2023/04/moe-for-recsys)
5. [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792)
6. [Towards the Better Ranking Consistency: A Multi-task Learning Framework for Early Stage Ads Ranking](https://arxiv.org/pdf/2307.11096)
7. [Practical Lessons from Predicting Clicks on Ads at Facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
8. [Attribution Modeling Increases Eﬃciency of Bidding in Display Advertising - Dataset](https://arxiv.org/pdf/1707.06409)
9. [Probability for Computer Scientists](https://chrispiech.github.io/probabilityForComputerScientists/en/)
10. [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/epdf/10.1145/3383313.3412236)
11. [DLRMv2 Implementation Github](https://github.com/mlcommons/inference/blob/master/recommendation/dlrm_v2/pytorch/README.md)
12. [Model Size Reduction Using Frequency Based Double Hashing for Recommender Systems](https://dl.acm.org/doi/10.1145/3383313.3412227)
13. [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531)