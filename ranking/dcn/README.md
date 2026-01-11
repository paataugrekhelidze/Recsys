### Deep & Cross Network (DCN)

DCN belongs to the family of models that automatically learn feature interactions, rather than relying on smart ML engineers to come up with them. Unlike DLRM, which only creates a second-order feature interactions, DCN allows for a bounded-degree orders, controlled by the number of stacked Cross Net layers. DCN consists of CrossNet and DNN layers, each trying to learn explicit and implicit feature interactions, respectively. CrossNet and DNN layers can either be stacked (f_cross * f_dnn) or parallel (f_cross+f_dnn), the architecture will mostly depend on the data and context.

Goals:
- Design a simple DCNv2 architecture and get similar performance on Criteo ads dataset as DLRM (AUC=0.80)
- Make DCNv2 more efficient by learning low-rank matrices in the crossNet layers
- Design Mixture of Experts (MoE) implementation of DCNv2, which extends on the low-rank matrix implementation and splits them into multiple experts within a single crossNet layer

Links:
- [Deep & Cross Network for Ad Click Predictions (DCNv1)](https://arxiv.org/pdf/1708.05123)
- [Improved Deep & Cross Network and Practical Lessons
for Web-scale Learning to Rank Systems (DCNv2)](https://arxiv.org/pdf/2008.13535)