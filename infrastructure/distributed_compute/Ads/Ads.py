import tempfile
from typing import Any

import torch
import datasets
import torch.nn as nn
from tricks.qr_embedding_bag import QREmbeddingBag
import math
import os
import time
# import ray
from ray.train import get_context, Checkpoint, report

class AdsDataset(torch.utils.data.Dataset):
    def __init__(self, data: datasets.arrow_dataset.Dataset,):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        
        # Return all features as a dictionary
        result = {}
        
        # Sparse features (scalars)
        sparse_features = ["uid", "campaign"] + [f"cat{i}" for i in range(1, 10)]
        for feat in sparse_features:
            result[feat] = torch.tensor(sample[feat], dtype=torch.long)
        
        # Position features (floats) - determine n_positions from your data
        n_positions = 174
        for i in range(n_positions):
            result[f"position_{i}"] = torch.tensor(sample[f"position_{i}"], dtype=torch.float32)
        
        # History features (variable length lists) - keep as python lists
        result["last_n_click_campaigns"] = sample["last_n_click_campaigns"]
        result["last_n_conversion_campaigns"] = sample["last_n_conversion_campaigns"]
        
        # Labels and metadata
        result["click"] = torch.tensor(sample["click"], dtype=torch.float32)
        result["conversion"] = torch.tensor(sample["conversion"], dtype=torch.float32)
        result["cost"] = torch.tensor(sample["cost"], dtype=torch.float32)
        result["timestamp"] = torch.tensor(sample["timestamp"], dtype=torch.long)
        
        return result
    
    # since history of clicks and conversions vary in length, there is an option for padding
    # but it is more practical to use 1D offset, that efficiently captures the whole batch and its length insensitive
    # it also work with embeddingBag and we do not need to worry about padded values. 
    @staticmethod
    def _collate_batch(batch: dict[str, Any], user_v: int, campaign_v: int) -> dict[str, torch.Tensor]:
        
        batched = {}
        for key, v in batch.items():
            if key in ["last_n_click_campaigns", "last_n_conversion_campaigns"]:
                action_list, action_offsets = [], [0]
                
                # Ray provides ragged arrays as arrays of objects (lists/arrays).
                # We can iterate through the batch dimension without converting the inner arrays to python lists.
                for sample in v:
                    # check if the sample (history) is empty
                    if len(sample) == 0:
                        processed_actions = torch.tensor([campaign_v], dtype=torch.int64)
                    else:
                        # Convert directly to tensor and apply vectorized modulo
                        processed_actions = torch.as_tensor(sample, dtype=torch.int64) % campaign_v
                    
                    action_list.append(processed_actions)
                    action_offsets.append(len(processed_actions))
                    
                # Concatenate into 1D embedding bag format
                batched[key+"_1D"] = torch.cat(action_list)
                # cumsum on offsets
                batched[key+"_1D_offset"] = torch.tensor(action_offsets[:-1], dtype=torch.int64).cumsum(dim=0)
                
            elif key == "uid":
                # Vectorized modulo across the entire batch natively
                batched[key] = torch.as_tensor(v, dtype=torch.int64) % user_v
                
            elif key == "campaign":
                # Vectorized modulo across the entire batch natively
                batched[key] = torch.as_tensor(v, dtype=torch.int64) % campaign_v
                
            else:
                # Default case for labels (click, conversion, cost) and categorical features
                # torch.as_tensor provides zero-copy conversion from NumPy arrays
                batched[key] = torch.as_tensor(v)
                
        return batched
    
class FinalRankerMMoE(nn.Module):
    """
    Multi-task multi-gate mixture of experts, similar to Youtube paper [3].
    Input data is transformed using a two-tower DLRM approach described in Meta paper [6].
    """
    def __init__(self, input_size = None, expert_num = 10, expert_dims = [512, 512], gate_dims = [], gate_dropout = None, task_dims = [[512, 256, 1], [512, 256, 1]], top_k = 3):
        super().__init__()
        self.expert_num = expert_num
        self.task_dims = task_dims
        self.expert_dims = expert_dims
        self.top_k = top_k
        self.gate_dropout = gate_dropout

        # initialize values for importance and load losses
        self.importance = None
        self.load = None

        # initialize experts and task gates (1 for each task)
        self.experts = self.init_experts(input_size = input_size, expert_num = expert_num, expert_dims = expert_dims)
        self.gates, self.noises = self.init_gates(input_size = input_size, task_num = len(task_dims), gate_dims = gate_dims, expert_num = expert_num, expert_dims = expert_dims)
        self.task_heads = self.init_heads(expert_dims = expert_dims, task_dims = task_dims)

        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.EmbeddingBag)):
            torch.nn.init.xavier_normal_(module.weight)



    def init_experts(self, input_size, expert_num, expert_dims):
        # within each layer element, multiple experts for the layer
        # each expert is just MLP, dimensions defined by expert_dims

        # Note: for each layer, do not include activations at the end of MLP
        experts = nn.ModuleList() # E x [MLP]
        for _ in range(expert_num):
            # Define MLP
            expert = nn.ModuleList()
            for d in range(len(expert_dims)):
                if d == 0:
                    expert.append(nn.Linear(in_features=input_size, out_features=expert_dims[d], bias=True))
                else:
                    expert.append(nn.Linear(in_features=expert_dims[d-1], out_features=expert_dims[d], bias=True))

                # add activation if not last layer of the MLP
                if d < len(expert_dims)-1:
                    expert.append(nn.ReLU())
            experts.append(nn.Sequential(*expert))

        return experts

    def init_heads(self, expert_dims, task_dims):
        # MLP for each task head
        task_heads = nn.ModuleList() # I x [MLP]
        for task in range(len(task_dims)):
            head = nn.ModuleList()
            for d in range(len(task_dims[task])):
                if d == 0:
                    head.append(nn.Linear(in_features=expert_dims[-1], out_features=task_dims[task][d], bias = True))
                else:
                    head.append(nn.Linear(in_features=task_dims[task][d-1], out_features=task_dims[task][d], bias = True))

                if d < len(task_dims[task])-1:
                    head.append(nn.ReLU())
            task_heads.append(nn.Sequential(*head))
        return task_heads
    

    def init_gates(self, input_size, task_num, gate_dims, expert_num, expert_dims):
        # create h gates, one for each task
        gates = nn.ModuleList()
        
        # noise per gate, if sparse MoE is implemented. Necessary to implement balanced traffic load across experts
        # instead of a deterministic 0/1 if above top_k, we define a distribution that we sample from, expected gate output(mean) + random_x * std, where random_x is sampled from N(0, I)
        # we basically learn normal distribution, mean and std, this allows us to define probability of > top_k using cdf of normal distribution, which allows us to calculate gradients, unlike the step function if we just used regular g_out > top_k
        # extra complexity but necessary to force the gates to distribute top_k "equally" across experts
        
        noises = nn.ModuleList()
        
        NUM_GATES = task_num
        for _ in range(NUM_GATES):
            gate = nn.ModuleList()
            for d in range(len(gate_dims)):
                if d == 0:
                    # define whether gate takes input from the input data or the previous expert layer
                    gate.append(nn.Linear(in_features=input_size, out_features=gate_dims[d], bias=False))
                else:
                    gate.append(nn.Linear(in_features=gate_dims[d-1], out_features=gate_dims[d], bias=False))
                
                if d < len(gate_dims)-1:
                    gate.append(nn.ReLU())

            # final projection layer to num_experts
            gate.append(nn.Linear(in_features=input_size if len(gate_dims) == 0 else gate_dims[-1], out_features=expert_num, bias=False))
            gates.append(nn.Sequential(*gate))

            # define noise weights
            # for simplicity I am just making a noise parameter a single learnable vector
            noise = nn.Linear(in_features=input_size, out_features=expert_num, bias=False)
            noises.append(noise)
    
        return gates, noises

    
    def gate_forward(self, x):
        # gate input is just a raw input x
        
        # (B, D) -> I x [(B, E)] where B is batch, D is input_dim, I separate gates for each task, E is a number of experts that goes in softmax

        # 3 separate strategies to balance expert utilization: importance loss, load loss (if sparse MoE), and gate dropout
        # importance loss tries to minimize variance of sum of softmax scores across experts per batch, so given a batch, if we add up scores from the gate softmax, total sum of weights across experts in a batch is forced to be similar
        # load loss tries to minimize variance of sum of probabilities of the softmax output > top_k across experts per batch, so each expert receives similar amount of probability of being > top_k
        # gate dropout randomly drops the scores before the softmax, preventing gate polarization (over-reliance on any specific expert)

        # list output for each task
        out = list() # I x [(B, E)]
        e_prob = list() # expert probabilities that their output (expected + variance) is > top_k across other experts
        runtime_device = x.device
        NUM_GATES = len(self.task_dims)
        for i in range(NUM_GATES):
            # Sparse MoE paper [1] equations 3, 4, 5
            mean = self.gates[i](x)
            routing_logits = mean
            # introduce noise only if sparsity is enabled
            # this technique is used only to balance # of times each expert is picked in top_k
            if self.top_k and self.top_k < self.expert_num:
                std = nn.functional.softplus(self.noises[i](x))
                noisy_logits = mean + torch.randn_like(input=std, device=runtime_device) * std

                top_values, _ = torch.topk(noisy_logits, k=self.top_k + 1, dim=1)
                k_largest = top_values[:, self.top_k-1].unsqueeze(-1) # (B, 1)
                k_next_largest = top_values[:, self.top_k].unsqueeze(-1) # (B, 1)

                # sampling from N(mean, std^2) represented as: mean + N(0, I)*std, where N(0, I) is a random sample from standard dist
                # keep only top_k
                # will be set to 0 when normalized in softmax
                # (B, E)
                routing_logits = noisy_logits.masked_fill(noisy_logits < k_largest, -float("inf"))


                # Sparse MoE paper [1] equations 8, 9
                # Goal of these equations is to define a load loss.
                # It tries to minimize variation between the sum of probabilities that the output is > top_k across experts
                # compute probability that H > top_k
                
                # needed to implement kth_excluding
                # threshold is defined as > kth_greatest, unless the expert is the kth_greatest, then threshold is moved to k+1th greatest
                # (B, E) - True where expert would be exactly kth
                is_kth_expert = (noisy_logits == k_largest)
                # (B, E) - threshold excluding each expert
                threshold_excluding = torch.where(is_kth_expert, k_next_largest, k_largest)
                
                # probability that, given the expected (mean) + noise (std) of the output from the gate for each sample on each expert, we will get the score that is > threshold
                # Insights from https://chrispiech.github.io/probabilityForComputerScientists/en/part2/normal/
                # z = (threshold - μ) / σ, where z score in N(0, I) distribution is equvalent to the threshold score in the N(μ, σ^2) distribution
                # P(Y <= threshold) = Φ(z)
                # P(Y > threshold) = 1 - Φ(z) = Φ(-z) = Φ((μ - threshold) / σ)
                
                # z = (threshold - μ) / σ
                z_score = (threshold_excluding - mean) / (std + 1e-9) # (B, E)
                # use precomputed cdf of standard normal
                # initialize standard normal dist, used to calculate probability for load loss
                std_normal = torch.distributions.Normal(
                    loc=torch.zeros((), device=z_score.device, dtype=z_score.dtype),
                    scale=torch.ones((), device=z_score.device, dtype=z_score.dtype),
                )
                # (B, E)
                i_prob = std_normal.cdf(-z_score) # Φ(-z)
                # I x [(B, E)]
                e_prob.append(i_prob)

            # (B, E)
            softmax = nn.functional.softmax(routing_logits, dim=1)
            # make sure this dropout is only applied during training, not during inference
            if self.training and self.gate_dropout:
                # Youtube paper [3] section 5.2.4 - Used to break expert polarization
                # Expert dropout: randomly zero out entire experts, then renormalize
                # Create dropout mask (B, E) where each expert has gate_dropout probability of being dropped
                expert_mask = torch.bernoulli(torch.full_like(input=softmax, fill_value=1 - self.gate_dropout))
                # Note: wanted to use toch.functional.dropout but it uses inverted drout, automatically scales remaining values by (1/(1-p)) to maintain the expected sum of inputs the same. Not useful in our case.
                softmax = softmax * expert_mask
                softmax = softmax / (torch.sum(input=softmax, dim=1, keepdim=True) + 1e-9) # re-normalize, prevent division by 0

            out.append(softmax)
            
        return out, e_prob
    
    def moe_forward(self, x):
        # MMoE implementation
        # (B, D) -> I x [(B, F)] where B is batch, D is input_dim, I separate outputs for each task, F is the last dimension of the experts

        # run gates for separate tasks
        # for each task, calculate weighted sum of experts
        # record sum of gate scores across batch for importance loss
        # record sum of gate probabilites of > kth_greatest across batch for load loss

        # importance and load variables
        importance = list()
        load = list()

        runtime_device = x.device

        # initialize zeros for the weighted sum across experts
        # I x [(B, M)] where M is the dimesion of the expert output vector, defined for I independent gates
        NUM_GATES = len(self.task_dims)
        gate_out = [torch.zeros((x.shape[0], self.expert_dims[-1]), device=runtime_device) for _ in range(NUM_GATES)]

        g, gate_probs = self.gate_forward(x=x) # outputs for all I gates
        # (B, E, F) B batch, E # of experts, F last dimension of each expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        for i in range(NUM_GATES):
            # calculate weighted sum from the ith gate

            # Youtube paper [3] equation 1
            # f(x) = sum(g_e(x) * f_e(x)) where e is the eth expert
            # sum((B, E, 1) * (B, E, F), dim=1) -> (B, F), where (B, E, 1) are scalar weights for E experts on B samples. (B, E, F) is F dimensional output for E experts on B samples.
            gate_out[i] = torch.sum(g[i].unsqueeze(-1) * expert_outputs, dim=1)
            
            # (B, E) -> (1, E) sum of gate score for each expert, summed by batch dimension
            importance.append(torch.sum(g[i], dim=0)) # ith gate

            # if sparsity is enabled, calculate load value for load loss
            if self.top_k and self.top_k < self.expert_num:
                # (B, E) -> (1, E) sum of gate probabilities for each expert score being > top_k, summed by batch dimension
                load.append(torch.sum(gate_probs[i], dim=0))


        x = gate_out

        return x, importance, load
    
    def task_forward(self, x):
        # I x [(B, D)] -> I x [(B, O)] - O dimensional output, for I separate tasks across B samples
        NUM_TASKS = len(self.task_dims)
        task_out = list()
        for task in range(NUM_TASKS):
            task_out.append(self.task_heads[task](x[task]))
        return task_out



    def forward(self, x):
        # (B, D) -> (B, T) where D is the input dimension, and T is the number of tasks

        # importance and load losses are per layer per task (basically 1 for each gate)
        self.importance = list() # I x [importance_i] where I is # of gates (equal to # of tasks), importance_i is importance of ith gate
        self.load = list()

        expert_out, importance, load = self.moe_forward(x)
        self.importance = importance
        self.load = load

        # Note: too many for loops, must start designing with model parallelism in mind
        # forward pass through each task to capture the task outputs
        # I x [(B, D)] -> I x [(B, 1)]
        task_out = self.task_forward(expert_out)
        
                
        return task_out
    
class EmbeddingLayer(nn.Module):
    """Modularize Embedding layer so the same tables can be shared across DLRMTowers"""
    def __init__(self, emb_layers, emb_dim):
        super().__init__()
        # initialize embedding tables for sparse features
        # Note: campaign feature needs a single embedding per sample (candidate id), 
        # last_n clicks and conversions need mean of embedding bag from the same table (history per sample)
        # no need to initialize embeddingbag tables for last_n clicks and conversions, they share it with the campaign feature
        self.embs = nn.ModuleDict()
        for emb_name, num_emb in emb_layers.items():
            # by default, quotient and remainder are combined with mult
            # by default, bag is aggregated with mean
            self.embs[emb_name] = QREmbeddingBag(num_categories=num_emb, embedding_dim=emb_dim, num_collisions=int(math.sqrt(num_emb)))

class DLRMTower(nn.Module):
    """
    Defines Bottom MLP + interaction layer for independent user and ad towers in early-stage model. 
    Creating user and ad embeddings on historic features independently will make 
    the architecture less expressive since we are not crossing features from the beginning (late interaction). The benefit of it
    is that it allows caching for batch features, which will improve inference latency for early-stage ranker. 
    If the constraints allow, this will not be used in the final ranker.
    """
    def __init__(self, bottom_mlp_layers, projection_layer, dense_num, sparse_num, embs):
        super().__init__()
        
        self.projection_layer = projection_layer
        self.embs = embs
            
        # initialize bottom mlp
        self.bottom_mlp = None
        if dense_num:
            self.bottom_mlp = nn.ModuleList()
            for i in range(len(bottom_mlp_layers)):
                self.bottom_mlp.append(nn.Linear(in_features=dense_num if i == 0 else bottom_mlp_layers[i-1], out_features=bottom_mlp_layers[i], bias=True))
                if i < len(bottom_mlp_layers)-1:
                    self.bottom_mlp.append(nn.ReLU())
            self.bottom_mlp = nn.Sequential(*self.bottom_mlp)

        # initialize projection layer
        # interaction performs matrix multiplication per sample: (N, D) x (D, N) -> (N, N)
        # N is the number of feature vectors: 1 (dense bottom MLP output) + sparse_num (embedding lookups)
        # we only need upper triangular, excluding the diagonal: z = N*(N-1) / 2
        N = sparse_num
        if dense_num:
            N += 1
        interaction_out = (N**2 - N) // 2
        # when dense is present, _interact concatenates the bottom MLP output (B, bottom_mlp_layers[-1]) with Z_flat
        # when dense is absent, _interact returns only Z_flat (B, interaction_out)
        proj_in = (bottom_mlp_layers[-1] if dense_num else 0) + interaction_out
        self.projection = nn.Linear(in_features=proj_in, out_features=projection_layer, bias=True)

        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.EmbeddingBag)):
            torch.nn.init.xavier_normal_(module.weight)

    def _interact(self, dense, sparse):
        """
        The following interact operation assumes only dot interactions between sparse and dense features
        Args:
            dense (torch.Tensor): dense input features of shape (B, D).
            sparse (list): list of K embedding lookups each of shape (B, D).
        Returns:
            Tensor output of shape (B, N) where N is the flattened upper triangular of second-order interactions.
        """
        # similar to factorization machine, combine features into a single matrix and run bmm against its transpose
        # get either upper or lower triangular since we do not need dup values
        
        B, D = sparse[0].shape
        # (B, (1+K)*D) -> (B, 1+K, D) if dense features present, else (B, K*D) -> (B, K, D)
        T = torch.cat([dense] + sparse if dense is not None else sparse, dim=1).view((B, -1, D))
        

        # (B, 1+K, D) x (B, D, 1+K) -> (B, 1+K, 1+K)
        Z = torch.bmm(T, torch.transpose(T, 1, 2))


        # get upper triangular for unique interactions, exlude diagonal
        row, col = torch.triu_indices(Z.shape[1], Z.shape[2], offset=1)
        # (B, 1+K, 1+K) -> (B, N) where N = (1+K)*(K) // 2
        Z_flat = Z[:, row, col]

        # combine original dense featues and flattened upper triangular of interactions
        # (B, N+D) if dense features present else (B, N)
        combined = torch.cat([dense, Z_flat], dim=1) if dense is not None else Z_flat

        return combined

    def forward(self, dense, emb_indices, emb_offsets):
        """
        Args:
            dense (torch.Tensor): dense input features.
            emb_indices dict[torch.Tensor]: embedding indices for k categories and B batch size -> (B+M, ) where M > 0 occurs for historic features.
            emb_offsets dict[torch.Tensor]: embedding offsets for k categories and B batch size -> (B, ).
        Returns:
            Tensor output of shape (B, P), where P is a projection dimension.
        """
        
        # step 1: score bottom MLP for dense features, skip if dense features not available.
        b_mlp_out = None
        if dense is not None:
            # (B, input) -> (B, D)
            b_mlp_out = self.bottom_mlp(dense)

        # not the prettiest way to get the batch size
        B = emb_indices["uid" if "uid" in emb_indices else "campaign"].shape[0]
        device = next(self.parameters()).device  # get model's device


        # step 2: embedding lookup across all sparse features
        emb_out = []
        for k in emb_indices.keys():
            # for historic features, reuse the campaign table
            emb_table = k if "last_n" not in k else "campaign"
            # for categorical features with single values, bag is always 1, so offset is just a range [0, B)
            offset = emb_offsets[k+"_offset"] if k+"_offset" in emb_offsets else torch.arange(start=0, end=B, device=device)
            emb_out.append(self.embs[emb_table](emb_indices[k], offset))
        
        # print(b_mlp_out.shape, len(emb_out))

        # step 3: calulate interaction matrix
        z = self._interact(b_mlp_out, emb_out)
        # print(z.shape)

        # projection
        return self.projection(z)
    
class EarlyRanker(nn.Module):
    """The following early-stage multi-task ranker architecture is inspired by the Meta paper [6]. It uses DLRM towers for 
    efficient user and ad feature transformation and caching, then defines a simply shared-bottom multi-task heads. Goal is to keep the
    model simple (fast) but expressive (accurate).
    
    Architecture:
    [DLRMTower(user); DLRMTower(ad); DLRMTower(context)] -> Shared-DNN -> Task Heads (CTR, CTR distill, CVR, CVR distill)
    """

    def __init__(self, shared_dims = [512, 256], task_dims = [[128, 64, 1], [128, 64, 1], [128, 64, 1], [128, 64, 1]], u_params = None, ad_params = None, c_params = None):
        """
        Initialize the early-stage ranker params
        Args:
            shared_dims (list): MLP layer dimensions for shared DNN.
            task_dims (list): MLP layer dimensions for task specific DNN.
            u_params (dict): dict specifying DLRMTower parameters for user features.
            ad_params (dict): dict specifying DLRMTower parameters for ad features.
            c_params (dict): dict specifying DLRMTower parameters for context features.
        """
        super().__init__()
        self.task_dims = task_dims
        shared_in = 0
        # initialize DLRM Towers
        if u_params:
            self.UTower = DLRMTower(**u_params)
            shared_in += self.UTower.projection_layer
        if ad_params:
            self.AdTower = DLRMTower(**ad_params)
            shared_in += self.AdTower.projection_layer
        if c_params:
            self.CTower = DLRMTower(**c_params)
            shared_in += self.CTower.projection_layer


        # initialize shared DNN
        self.shared = nn.ModuleList()
        for i in range(len(shared_dims)):
            self.shared.append(nn.Linear(in_features=shared_in if i == 0 else shared_dims[i-1], out_features=shared_dims[i], bias=True))
            if i < len(shared_dims)-1:
                self.shared.append(nn.ReLU())
        self.shared = nn.Sequential(*self.shared) # in_features = M*P, out_features =  D, where M is number of DLRM towers, P is projection dim, D is output dim of shared DNN

        # initialize task heads
        self.task_heads = nn.ModuleList() # I x [MLP]
        for task in range(len(task_dims)):
            head = nn.ModuleList()
            for i in range(len(task_dims[task])):
                head.append(nn.Linear(in_features=shared_dims[-1] if i == 0 else task_dims[task][i-1], out_features=task_dims[task][i], bias = True))
                if i < len(task_dims[task])-1:
                    head.append(nn.ReLU())
            self.task_heads.append(nn.Sequential(*head))
        
        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.EmbeddingBag)):
            torch.nn.init.xavier_normal_(module.weight)
    
    def forward(self, cache = None, x = None):
        """forward pass takes advantage of cached values (e.g. user and ad embeddings) if available
        Args:
            cache (dict[str, torch.Tensor] | None): dict of cached DLRMTower output tensors of shape (B, P),
                where P is the projection dim of a tower. Keys: "user", "ad", "context".
            x (dict[str, dict]): dict keyed by tower name ("user", "ad", "context"), each value is a dict with:
                - "dense"       (torch.Tensor | None): dense input of shape (B, M), or None if no dense features.
                - "emb_indices" (dict[str, torch.Tensor]): sparse feature indices;
                                  scalar features -> (B,), historic bag features -> (B+M,) flattened 1D, where M >= 0.
                - "emb_offsets" (dict[str, torch.Tensor]): cumulative bag offsets for historic features -> (B,).
                                  scalar features do not need entries here (arange fallback used in DLRMTower).
        Returns:
            list[torch.Tensor]: T tensors of shape (B, D), one per task head.
        """

        # use cached DLRMTower Embeddings
        user = cache["user"] if cache is not None and "user" in cache else None
        ad = cache["ad"] if cache is not None and "ad" in cache else None
        context = cache["context"] if cache is not None and "context" in cache else None

        if x is not None:
            # calculate DLRMTower Embeddings if cache is missing
            if user is None and "user" in x:
                user = self.UTower(**x["user"])
            if ad is None and "ad" in x:
                ad = self.AdTower(**x["ad"])
            if context is None and "context" in x:
                context = self.CTower(**x["context"])
        
        combined = list()
        if user is not None:
            combined.append(user)
        if ad is not None:
            combined.append(ad)
        if context is not None:
            combined.append(context)
        
        combined = torch.cat(combined, dim=1) # (B, P_u + P_ad + P_c)

        # Forward shared DNN
        shared_out = self.shared(combined)

        # Forward task heads
        NUM_TASKS = len(self.task_dims)
        task_out = list()
        for task in range(NUM_TASKS):
            task_out.append(self.task_heads[task](shared_out))
        
        return task_out
    

class FullRanker(nn.Module):
    """
    Wraps DLRMTowers + FinalRankerMMoE into a single module compatible with Solver.
    Towers produce per-tower projections, concatenated and fed into the final ranker.
    """
    def __init__(self, towers: dict, final_ranker: FinalRankerMMoE):
        """
        Args:
            towers (dict[str, DLRMTower]): keyed by tower name e.g. {"user": uTower, "ad": adTower}
            final_ranker (FinalRankerMMoE): pre-initialized final ranker
        """
        super().__init__()
        self.towers = nn.ModuleDict(towers)
        self.final_ranker = final_ranker
        # expose these so Solver's sparse MoE loss check works unchanged
        self.top_k = final_ranker.top_k
        self.expert_num = final_ranker.expert_num

    @property
    def importance(self):
        return self.final_ranker.importance

    @property
    def load(self):
        return self.final_ranker.load

    def forward(self, x):
        """
        Args:
            x (dict[str, dict]): keyed by tower name, each value is a dict with:
                - "dense"       (torch.Tensor | None)
                - "emb_indices" (dict[str, torch.Tensor])
                - "emb_offsets" (dict[str, torch.Tensor])
        Returns:
            list[torch.Tensor]: T tensors of shape (B, 1), one per task head.
        """
        tower_outs = []
        for name, tower in self.towers.items():
            tower_outs.append(tower(**x[name]))

        # (B, P_u + P_ad + ...)
        combined = torch.cat(tower_outs, dim=1)
        return self.final_ranker(combined)

class Solver:
    def __init__(self, 
                 model, 
                 data,
                 optimizer,
                 epochs,
                 batch_size,
                 collate_fn,
                 checkpoint_dir = "./checkpoints",
                 checkpoint_every = 1, 
                 verbose = True, 
                 print_every = 1, 
                 device = "cpu",
                #  reset = False,
                 importance_weight = 0,
                 load_weight = 0,
                 distillation_weights = [],
                 distillation_tasks = [],
                 teacher = None,
                 start_epoch = 0
                ):
        self.model = model
        self.data = data
        self.epochs = epochs
        self.verbose = verbose
        # self.reset = reset
        self.print_every = print_every
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every  
        self.optimizer = optimizer
        self.importance_weight = importance_weight
        self.load_weight = load_weight
        self.distillation_weights = distillation_weights
        self.distillation_tasks = distillation_tasks
        self.teacher = teacher
        self.device = device
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.start_epoch = start_epoch

        # self.loss_history = []
        # self.loss_history_batch = []
        # self.load_cv_history_batch = []
        # self.importance_cv_history_batch = []



    @staticmethod
    def _build_x(batch):
        """Construct the structured x dict expected by FullRanker.forward from a flat collated batch dict.
        Override this method if your tower/feature split differs.
        Args:
            batch (dict[str, torch.Tensor]): flat collated batch from AdsDataset._collate_batch.
        Returns:
            tuple[dict, list[torch.Tensor]]: x dict for model, list of target tensors per task.
        """
        x = {
            "user": {
                "dense": None,
                "emb_indices": {
                    "uid": batch["uid"],
                    "last_n_click_campaigns_1D": batch["last_n_click_campaigns_1D"],
                    "last_n_conversion_campaigns_1D": batch["last_n_conversion_campaigns_1D"],
                },
                "emb_offsets": {
                    "last_n_click_campaigns_1D_offset": batch["last_n_click_campaigns_1D_offset"],
                    "last_n_conversion_campaigns_1D_offset": batch["last_n_conversion_campaigns_1D_offset"],
                },
            },
            "ad": {
                "dense": None,
                "emb_indices": {k: batch[k] for k in ["campaign"] + [f"cat{i}" for i in range(1, 10)]},
                "emb_offsets": {},
            },
        }
        # one target tensor per task head, shape (B, 1)
        targets = [
            batch["click"].float().unsqueeze(1),
            batch["conversion"].float().unsqueeze(1),
        ]
        return x, targets

    def _step(self):
        total_loss = 0
        # nbatches = len(self.data)
        data_iter = self.data.iter_torch_batches(
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=False,
            prefetch_batches=0 # do not prefetch more due to OOM
        )

        for counter, batch in enumerate(data_iter):
            # print(f"[{counter}/{nbatches}]")

            # move all tensors in the collated batch dict to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            x, targets = self._build_x(batch)

            # skips memset to zero op and makes training slightly faster
            self.optimizer.zero_grad(set_to_none=True)

            # FullRanker.forward(x) -> T x [(B, 1)], one tensor per task
            task_logits = self.model(x = x)

            # sum BCE loss across all non-distillation task heads
            loss = sum(
                nn.functional.binary_cross_entropy_with_logits(
                    input=task_logits[t],
                    target=targets[t],
                    reduction="mean",
                )
                for t in range(len(task_logits)-len(self.distillation_tasks))
            )

            if self.teacher is not None:
                with torch.no_grad():
                    teacher_task_logits = self.teacher(x = x)
                if len(self.distillation_tasks):
                    if len(self.distillation_weights) == 0:
                        self.distillation_weights = [1] * len(self.distillation_tasks)
                    
                    # strategy 2: meta paper [6] equation 4
                    # Separate distillation tasks (4 tasks total: CTR, CVR, CTR_distill, CVR_distill)
                    # distillation_tasks = [2, 3] means task 2 learns from teacher task 0, task 3 from teacher task 1

                    for i, student_task_idx in enumerate(self.distillation_tasks):
                        # student_task_idx is the task head index in the student model (e.g., 2 or 3)
                        # i is the corresponding teacher task index (0 or 1)
                        loss += nn.functional.binary_cross_entropy_with_logits(
                            input=task_logits[student_task_idx],
                            target=torch.sigmoid(teacher_task_logits[i]),  # Use i, not student_task_idx
                            reduction="mean",
                        ) * self.distillation_weights[i]
                    
                else:
                    if len(self.distillation_weights) == 0:
                        self.distillation_weights = [1] * len(task_logits)
                    # strategy 1: distillation paper [13] section 2
                    # Same tasks with combined hard + soft targets (2 tasks: CTR, CVR)
                    # Each task learns from BOTH ground truth (already computed above) AND teacher soft targets
                    
                    # Add soft target loss for each task (already has hard target loss from earlier)
                    for t in range(len(task_logits)):
                        loss += nn.functional.binary_cross_entropy_with_logits(
                            input=task_logits[t],
                            target=torch.sigmoid(teacher_task_logits[t]),
                            reduction="mean",
                        ) * self.distillation_weights[t]

            
            underlying = self.model.module if hasattr(self.model, "module") else self.model
            # sparse MoE auxiliary losses (importance + load), only when sparsity is active
            is_sparse = hasattr(underlying, "top_k") and hasattr(underlying, "expert_num") \
                        and underlying.top_k and underlying.top_k < underlying.expert_num
            if is_sparse:
                # self.model.importance: L x [I x (E,)]
                # CV is computed per gate (per layer, per expert group) so each gate is independently
                # pushed toward uniform expert utilization. Aggregating across layers before std/mean
                # would conflate gates operating at different scales with independent routing behaviors.
                # Sparse MoE paper [3] equation 7
                importance_loss = torch.tensor(0.0, device=self.device)
                importance_cv_sum = 0.0
                n_gates = 0
                for gate_importance in underlying.importance:  # gate_importance: (E,)
                    std, mean = torch.std_mean(gate_importance)
                    cv = std / (mean + 1e-9)
                    importance_loss = importance_loss + cv ** 2
                    importance_cv_sum += float(cv.item())
                    n_gates += 1
                importance_loss = self.importance_weight * importance_loss

                # Sparse MoE paper [3] equation 11
                load_loss = torch.tensor(0.0, device=self.device)
                load_cv_sum = 0.0
                for gate_load in underlying.load:  # gate_load: (E,)
                    std, mean = torch.std_mean(gate_load)
                    cv = std / (mean + 1e-9)
                    load_loss = load_loss + cv ** 2
                    load_cv_sum += float(cv.item())
                load_loss = self.load_weight * load_loss

                loss = loss + importance_loss + load_loss


            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            # self.loss_history_batch.append(float(loss.item()))

        return total_loss / max(counter+1, 1)

    
    def _save(self, model, loss, epoch, filepath):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint = {
            "epoch" : epoch,
            "loss" : loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at {filepath}")

    def _load(self, filepath):
        if not os.path.exists(filepath):
            return None, None
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        return epoch, loss
    
    def train(self):
        
        # Set the student model to training mode
        self.model.train()
        if self.teacher is not None:
            # set the teacher model to eval mode
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start_time = time.time()
            loss = self._step()
            epoch_end_time = time.time()
            
            if self.verbose and epoch % self.print_every == 0:
                print(f"[{epoch}/{self.epochs}] Loss: {loss:.6f} time: {(epoch_end_time - epoch_start_time) / 60.0}m")
            
            metrics = {"loss": loss, "epoch": epoch}
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None

                try:
                    world_rank = get_context().get_world_rank()
                except Exception:
                    world_rank = 0  # fallback for local/non-Ray execution

                # Only the global rank 0 (head/master) saves the checkpoint
                if world_rank == 0:
                    model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                    self._save(model = model_to_save, loss = loss, epoch = epoch, filepath = os.path.join(temp_checkpoint_dir, f"model.pt"))
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                report(metrics, checkpoint=checkpoint)

            # self.loss_history.append(loss)