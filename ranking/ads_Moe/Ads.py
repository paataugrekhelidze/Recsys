import torch
import datasets
import torch.nn as nn
from tricks.qr_embedding_bag import QREmbeddingBag
import math

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
    def _collate_batch(batch, user_v, campaign_v):
        batched = {}
        for key in batch[0].keys():
            if key in ["last_n_click_campaigns", "last_n_conversion_campaigns"]:
                action_list, action_offsets = [], [0]
                for sample in batch:
                    actions = sample[key]

                    if not actions:
                        # represents a missing campaign embedding, let the model learn what the value should represent
                        # some users may not have any recent clicks or conversions, so their embedding will be represented with a learnable null vector
                        # unknown ids naturally mapped to a controllable hash space, which also helps with the cold-start problem
                        # alternatively, locality based hashing can be implemented to better deal with cold-start problem, e.g. same hash for ads with similar attributes (same advertiser, topic, marketing type...)
                        # NOTE: campaign_v is used as the reserved null index; embedding table must have size campaign_v + 1
                        processed_actions = torch.tensor([campaign_v], dtype=torch.int64)
                    else:
                        # hash real campaign ids into [0, campaign_v - 1], keeping campaign_v free for the null sentinel
                        processed_actions = torch.as_tensor(actions, dtype=torch.int64) % campaign_v
                    
                    action_list.append(processed_actions)
                    action_offsets.append(len(processed_actions))
                # click and conversion list represented as 1D; real ids in [0, campaign_v-1], null sentinel at campaign_v
                batched[key+"_1D"] = torch.cat(action_list)
                batched[key+"_1D_offset"] = torch.tensor(action_offsets[:-1]).cumsum(dim=0) # tensor([13,  7,  3, 10, 13,  3, 15, 10,  9, 10]) -> tensor([13, 20, 23, 33, 46, 49, 64, 74, 83, 93])
            elif key == "uid":
                batched[key] = torch.stack([sample[key] % user_v for sample in batch])
            elif key == "campaign":
                batched[key] = torch.stack([sample[key] % campaign_v for sample in batch])
            else:
                # stack other labels as usual
                batched[key] = torch.stack([sample[key] for sample in batch])
        
        return batched
    
class FinalRankerMMoE(nn.Module):
    """
    Multi-layer Multi-task multi-gate mixture of experts (ML-MMoE), similar to Youtube paper [3]. Defaults to MMoE if layers = 1
    Input data is transformed using a two-tower DLRM approach described in Meta paper [6].
    """
    def __init__(self, input_size = None, layers = 1, expert_num = 10, expert_dims = [512, 512], gate_dims = [], gate_dropout = None, task_dims = [[512, 256], [512, 256]], top_k = 3, device = "cpu"):
        super().__init__()
        self.layers = layers
        self.expert_num = expert_num
        self.task_dims = task_dims
        self.expert_dims = expert_dims
        self.top_k = top_k
        self.device = device
        self.gate_dropout = gate_dropout

        # initialize values for importance and load losses
        self.importance = None
        self.load = None
        
        # initialize standard normal dist, used to calculate probability for load loss
        self.std_normal = torch.distributions.Normal(loc=0., scale=1.)

        # initialize experts for each layer and gates (1 for each task and layer)
        self.experts = self.init_experts(input_size = input_size, layers = layers, expert_num = expert_num, expert_dims = expert_dims, device = device)
        self.gates, self.noises = self.init_gates(input_size = input_size, layers = layers, task_num = len(task_dims), gate_dims = gate_dims, expert_num = expert_num, expert_dims = expert_dims)
        self.task_heads = self.init_heads(expert_dims = expert_dims, task_dims = task_dims)



    def init_experts(self, input_size, layers, expert_num, expert_dims, device):
        # l elements, each representing a layer
        # within each layer element, multiple experts for the layer
        # each expert is just MLP, dimensions defined by expert_dims

        # Note: for each layer, do not include activations at the end of MLP
        # makes sense to do on MLPs belonging to the last layer, but why apply the same rule to the intermediate layers
        # more expressive power to the weighted sum???
        experts = nn.ModuleList() # L x [ E x [MLP]]
        for l in range(layers):
            layer_experts = nn.ModuleList() # experts for layer l
            for _ in range(expert_num):
                # Define MLP
                expert = nn.ModuleList()
                for d in range(len(expert_dims)):
                    if d == 0:
                        if l == 0:
                            expert.append(nn.Linear(in_features=input_size, out_features=expert_dims[d], bias=True, device=device))
                        else:
                            # last dimension of the previous layer's expert should be the input dimension of next layer's expert
                            expert.append(nn.Linear(in_features=expert_dims[-1], out_features=expert_dims[d], bias=True, device=device))
                    else:
                        expert.append(nn.Linear(in_features=expert_dims[d-1], out_features=expert_dims[d], bias=True, device=device))

                    # add activation if not last layer of the MLP
                    if d < len(expert_dims)-1:
                        expert.append(nn.ReLU())
                layer_experts.append(nn.Sequential(*expert))
            experts.append(layer_experts)

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
    

    # MUST BE UPDATED, CURRENTLY EACH LAYER ONLY HAS GATES FOR PARTICULAR TASK
    # INSTEAD WE NEED GATES FOR EACH EXPERT IN NEXT LAYER, I THINK...
    def init_gates(self, input_size, layers, task_num, gate_dims, expert_num, expert_dims):
        # for each layer l, create h gates, one for each expert or task in next layer
        gates = nn.ModuleList()
        
        # noise per gate, if sparse MoE is implemented. Necessary to implement balanced traffic load across experts
        # instead of a deterministic 0/1 if above top_k, we define a distribution that we sample from, expected gate output(mean) + random_x * std, where random_x is sampled from N(0, I)
        # we basically learn normal distribution, mean and std, this allows us to define probability of > top_k using cdf of normal distribution, which allows us to calculate gradients, unlike the step function if we just used regular g_out > top_k
        # extra complexity but necessary to force the gates to distribute top_k "equally" across experts
        
        noises = nn.ModuleList()
        for l in range(layers):
            # h gates for a specific layer l, each corresponding to a task
            layer_gates = nn.ModuleList()
            layer_noises = nn.ModuleList()
            NUM_GATES = task_num if l == layers-1 else expert_num
            for _ in range(NUM_GATES):
                gate = nn.ModuleList()
                for d in range(len(gate_dims)):
                    if d == 0:
                        # define whether gate takes input from the input data or the previous expert layer
                        if l == 0:
                            gate.append(nn.Linear(in_features=input_size, out_features=gate_dims[d], bias=False))
                        else:
                            gate.append(nn.Linear(in_features=expert_dims[-1], out_features=gate_dims[d], bias=False))
                    else:
                        gate.append(nn.Linear(in_features=gate_dims[d-1], out_features=gate_dims[d], bias=False))
                    
                    if d < len(gate_dims)-1:
                        gate.append(nn.ReLU())

                # final projection layer to num_experts
                if l == 0:
                    gate.append(nn.Linear(in_features=input_size if len(gate_dims) == 0 else gate_dims[-1], out_features=expert_num, bias=False))
                else:
                    gate.append(nn.Linear(in_features=expert_dims[-1] if len(gate_dims) == 0 else gate_dims[-1], out_features=expert_num, bias=False))
                layer_gates.append(nn.Sequential(*gate))

                # define noise weights
                # for simplicity I am just making it a single learnable layer
                # within a layer, there as many gates as # of heads
                if l == 0:
                    noise = nn.Linear(in_features=input_size, out_features=expert_num, bias=False)
                else:
                    noise = nn.Linear(in_features=expert_dims[-1], out_features=expert_num, bias=False)
                layer_noises.append(noise)

            gates.append(layer_gates)
            noises.append(layer_noises)
        
    
        return gates, noises

    
    def gate_forward(self, x, layer):
        # gate input is raw input at layer 0 OR weighted sum output of the associated gate in the previous layer. PLE paper [10] equation 6
        
        # (B, D) or I x [(B, E)] -> I x [(B, E)] where B is batch, D is input_dim or last dim of last layers' expert, I separate gates (each for separate experts in next layer), E is a number of experts that goes in the input



        # 3 separate strategies to balance expert utilization: importance loss, load loss (if sparse MoE), and gate dropout
        # importance loss tries to minimize variance of sum of softmax scores across experts per batch, so given a batch, if we add up scores from the gate softmax, total sum of weights across experts in a batch is forced to be similar
        # load loss tries to minimize variance of sum of probabilities of the softmax output > top_k for a given expert across the batch, so each expert receive similar amount of probability of being > top_k
        # gate dropout randomly drops the scores before the softmax, preventing gate polarization (over-reliance on specific experts)

        # list output for each task
        out = list() # I x [(B, E)]
        e_prob = list() # expert probabilities that their output (expected + variance) is > top_k across other experts
        NUM_GATES = self.expert_num if layer < self.layers-1 else len(self.task_dims)
        for i in range(NUM_GATES):
            # Sparse MoE paper [1] equations 3, 4, 5
            mean = self.gates[layer][i](x) if layer == 0 else self.gates[layer][i](x[i]) # PLE paper [10] equation 6, progressive routing
            # (B, E)
            H_l = mean
            # introduce noise only if sparsity is enabled
            # this technique is used only to balance # of times each expert is picked in top_k
            if self.top_k and self.top_k < self.expert_num:
                std = nn.functional.softplus(self.noises[layer][i](x))
                # sampling from N(mean, std^2) represented as: mean + N(0, I)*std, where N(0, I) is a random sample from standard dist
                H_l += torch.randn_like(input=std, device=self.device) * std
            
                # keep only top_k
                k_largest, _ = torch.kthvalue(-H_l, self.top_k, dim=1)
                k_largest = -k_largest.unsqueeze(-1) # (B, 1)
                # will be set to 0 when normalized in softmax
                # (B, E)
                H_l[H_l < k_largest] = -float("inf")


                # Sparse MoE paper [1] equations 8, 9
                # Goal of these equations is to define a load loss.
                # It tries to minimize variation between the sum of probabilities that the output is > top_k across experts
                # compute probability that H_l > top_k
                
                # needed to implement kth_excluding
                # threshold is defined as > kth_greatest, unless the expert is the kth_greatest, then threshold is moved to k+1th greatest
                k_next_largest, _ = torch.kthvalue(-H_l, self.top_k+1, dim=1)
                k_next_largest = -k_next_largest.unsqueeze(-1)

                # (B, E) - True where expert would be exactly kth
                is_kth_expert = (H_l == k_largest)
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
                # (B, E)
                i_prob = self.std_normal.cdf(-z_score) # Φ(-z)
                # I x [(B, E)]
                e_prob.append(i_prob)

            # (B, E)
            softmax = nn.functional.softmax(H_l, dim=1)
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
        # ML-MMOE implementation, default to MMOE if layers = 1
        # (B, D) -> I x [(B, E)]

        # iterate through each layer
        # within each layer, run gates for separate tasks
        # for each task, calculate weighted sum of experts in each layer
        # record sum of gate scores across batch for importance loss
        # record sum of gate probabilites of > kth_greatesd across batch for load loss

        importance = list()
        load = list()

        x_l = x # input at layer l
        for l in range(self.layers):
            # initialize layer specific importance and load variables
            layer_importance = list() # I x [importance_l_i]
            layer_load = list()

            # initialize zeros for the weighted sum across experts for a given layer
            # I x [(B, M)] where M is the dimesion of the expert output vector, defined for I independent gates
            NUM_GATES = self.expert_num if l < self.layers-1 else len(self.task_dims)
            layer_gate_out = [torch.zeros((x.shape[0], self.expert_dims[-1]), device=self.device)] * NUM_GATES

            g, gate_probs = self.gate_forward(x=x_l, layer=l) # outputs for all I gates at layer l
            for i in range(NUM_GATES):
                # calculate weighted sum from the ith gate

                # Youtube paper [3] equation 1
                # f(x) = sum(g_e(x) * f_e(x)) where e is the eth expert
                for e in range(self.expert_num):
                    # expert input is raw input at layer 0 OR weighted sum output of the associated gate in the previous layer. PLE paper [10] equation 6
                    # (B, M) is the M dimenional output of the eth expert across B batches
                    f_e = self.experts[l][e](x_l) if l == 0 else self.experts[l][e](x_l[e])
                    # (B, 1) * (B, M)
                    # (B, 1) are weights from the gate for eth expert across B batches
                    layer_gate_out[i] += g[i][:, e].unsqueeze(-1) * f_e
                
                # (B, E) -> (1, E) sum of gate score for each expert, summed by batch dimension
                importance_l_i = torch.sum(g[i], dim=0) # ith gate at layer l
                layer_importance.append(importance_l_i)

                # if sparsity is enabled, calculate load value for load loss
                if self.top_k and self.top_k < self.expert_num:
                    # (B, E) -> (1, E) sum of gate probabilities for each expert score being > top_k, summed by batch dimension
                    load_l_i = torch.sum(gate_probs[i], dim=0)
                    layer_load.append(load_l_i)

            importance.append(layer_importance)
            # if sparsity is enabled, calculate load value for load loss
            if self.top_k and self.top_k < self.expert_num:
                load.append(layer_load)

            # .... must figure out x_l for next layer, if next layer experts get weighted sum of previous experts weighted by a unique gate, then what should the next layer gate network take as input???
            x_l = layer_gate_out

        return x_l, importance, load
    
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
        self.importance = list() # L x [ I x [importance_l_i]] where L is layer, I is # of gates at layer l (equal to # of tasks at the last layer, otherwise # of experts in next layer), importance_l_i is importance of ith gate at layer l 
        self.load = list()

        expert_out, importance, load = self.moe_forward(x)
        self.importance = importance
        self.load = load

        # Note: too many for loops, must start designing with model parallelism in mind
        # forward pass through each task to capture the task outputs
        # I x [(B, D)] -> I x [(B, 1)]
        task_out = self.task_forward(expert_out)
        
                
        return task_out
    
class DLRMTower(nn.Module):
    """
    Defines Bottom MLP + interaction layer for independent user and ad towers in early-stage model. 
    Creating user and ad embeddings on historic features independently will make 
    the architecture less expressive since we are not crossing features from the beginning (late interaction). The benefit of it
    is that it allows caching for batch features, which will improve inference latency for early-stage ranker. 
    If the constraints allow, this will not be used in the final ranker.
    """
    def __init__(self, bottom_mlp_layers, projection_layer, dense_num, sparse_num, emb_layers, device):
        super().__init__()
        self.device = device

        # initialize embedding tables for sparse features
        # Note: campaign feature needs a single embedding per sample (candidate id), 
        # last_n clicks and conversions need mean of embedding bag from the same table (history per sample)
        # no need to initialize embeddingbag tables for last_n clicks and conversions, they share it with the campaign feature
        self.embs = nn.ModuleDict()
        for emb_name, num_emb in emb_layers.items():
            # by default, quotient and remainder are combined with mult
            # by default, bag is aggregated with mean
            self.embs[emb_name] = QREmbeddingBag(num_categories=num_emb, embedding_dim=bottom_mlp_layers[-1], num_collisions=int(math.sqrt(num_emb)))
            
        
        # initialize bottom mlp
        self.bottom_mlp = None
        if dense_num:
            self.bottom_mlp = nn.ModuleList()
            for i in range(len(bottom_mlp_layers)):
                self.bottom_mlp.append(nn.Linear(in_features=dense_num if i == 0 else bottom_mlp_layers[i-1], out_features=bottom_mlp_layers[i], bias=True, device=device))
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
        self.projection = nn.Linear(in_features=proj_in, out_features=projection_layer, bias=True, device=device)

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

    def forward(self, x, emb_indices, emb_offsets):
        """
        Args:
            x (torch.Tensor): dense input features.
            emb_indices dict[torch.Tensor]: embedding indices for k categories and B batch size -> (B+M, ) where M > 0 occurs for historic features.
            emb_offsets dict[torch.Tensor]: embedding offsets for k categories and B batch size -> (B, ).
        Returns:
            Tensor output of shape (B, P), where P is a projection dimension.
        """
        
        # step 1: score bottom MLP for dense features, skip if dense features not available.
        b_mlp_out = None
        if x is not None:
            # (B, input) -> (B, D)
            b_mlp_out = self.bottom_mlp(x)

        B = emb_offsets["last_n_click_campaigns_1D_offset"].shape[0]
        # step 2: embedding lookup across all sparse features
        emb_out = []
        for k in emb_indices.keys():
            # for historic features, reuse the campaign table
            emb_table = k if "last_n" not in k else "campaign"
            # for categorical features with single values, bag is always 1, so offset is just a range [0, B)
            offset = emb_offsets[k+"_offset"] if k+"_offset" in emb_offsets else torch.arange(start=0, end=B, device=self.device)
            emb_out.append(self.embs[emb_table](emb_indices[k], offset))
        
        # print(b_mlp_out.shape, len(emb_out))

        # step 3: calulate interaction matrix
        z = self._interact(b_mlp_out, emb_out)
        # print(z.shape)

        # projection
        return self.projection(z)