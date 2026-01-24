import torch.nn as nn
import torch
import math
from tricks.qr_embedding_bag import QREmbeddingBag
import os
import time

class CrossNet(nn.Module):
    def __init__(self, input_size, layers = 4, low_rank = False, r = 256, v1 = False, gate_layers = [], num_experts = 1, top_k = 1):
        super().__init__()
        self.input_size = input_size
        self.layers = layers
        self.low_rank = low_rank
        self.r = r
        self.v1 = v1
        self.gate_layers = gate_layers
        self.num_experts = num_experts
        self.top_k = top_k
        # track batchwise sum of gate output for each expert
        # will be used to minimize variance across experts, so that each expert gets roughly the same amount of batchwise sum of gate scores
        # this translates to similar amount of gradients per expert, even if the samples per expert vary
        self.importance = None
        # importance loss is not enough, it is possible that we satisfy the requirements of balanced 
        # batchwise sum of gate scores across experts
        # but there might be a case where an expert gets average scores where none of them reach top-k, we now need to also enforce
        # balanced top-k across experts
        self.load = None

        # Store experts in nn.ModuleList so parameters are properly registered
        self.e_models = nn.ModuleList()

        # initialize standard normal dist, used to calculate probability for load loss
        self.std_normal = torch.distributions.Normal(loc=0., scale=1.)

        if not low_rank:
            for _ in range(layers):
                steps = nn.ModuleList()
                for _ in range(num_experts):
                    if not v1:
                        # x𝑙+1 = x0 ⊙ (𝑊𝑙x𝑙 +b𝑙)+x𝑙 - initialize 𝑊𝑙 and b𝑙 - DCNv2 Paper [2] equation 1
                        # full matrix
                        steps.append(nn.Linear(in_features=input_size, out_features=input_size, bias=True))
                    else:
                        # x𝑙+1 = x0 * (x𝑙^T @ w𝑙) + b𝑙 + x𝑙 - initialize w𝑙 and b𝑙 - DCNv1 Paper [1] equation 3
                        # rank 1 vector
                        steps.append(nn.Linear(in_features=input_size, out_features=1, bias=True))
                # list of l layers, each layer list contains a list of experts
                self.e_models.append(steps)
        else:
            # x𝑙+1 = x0 ⊙ (𝑈𝑙(𝑉𝑙^T @ x𝑙) + b𝑙) + x𝑙 - initialize 𝑈𝑙, 𝑉𝑙, and b𝑙 - DCNv2 Paper [2] equation 2
            # rank_r matrix
            rank = max(1, r)
            weight_shape = (input_size, rank)
            bias_shape = (input_size,)
            for _ in range(layers):
                steps = nn.ModuleList()
                for _ in range(num_experts):
                    U = nn.Parameter(torch.empty(weight_shape))
                    V = nn.Parameter(torch.empty(weight_shape))
                    b = nn.Parameter(torch.empty(bias_shape))
                    steps.append(nn.ParameterDict({
                        "U": U,
                        "V": V,
                        "bias": b
                    }))
                self.e_models.append(steps)

        self.apply(self._init_weights)

        # Sparse MoE paper [3] equations 3 & 4
        if num_experts > 1:
            g_steps = nn.ModuleList()
            # create multi-layered gate if specified
            for i in range(len(gate_layers)):
                step = nn.Linear(in_features=input_size if i == 0 else gate_layers[i-1], out_features=gate_layers[i], bias=False)

                # Initialize weights to zero for W_g
                # "To avoid out-of-memory errors, we need to initialize the network in a
                # state of approximately equal expert load (since the soft constraints need some time to work). To
                # accomplish this, we initialize the matrices Wg and Wnoise to all zeros, which yields no signal and
                # some noise."
                torch.nn.init.zeros_(step.weight)

                g_steps.append(step)
                g_steps.append(nn.ReLU())

            # last, projection layer that projects to num_experts
            step = nn.Linear(in_features=gate_layers[i-1] if len(gate_layers) else input_size, out_features=num_experts, bias=False)
            
            # Initialize weights to zero for W_g
            torch.nn.init.zeros_(step.weight)
            
            g_steps.append(step)
            self.g_model = nn.Sequential(*g_steps)

            self.g_noise = nn.Linear(in_features=input_size, out_features=num_experts, bias=False)
            # Initialize weights to zero for W_noise
            torch.nn.init.zeros_(self.g_noise.weight)



    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.ParameterDict):
            # Initialize custom parameters in ParameterDict
            for param in module.values():
                if param.dim() >= 2:
                    # U and V matrices
                    torch.nn.init.xavier_normal_(param)
                else:
                    # bias vector
                    torch.nn.init.zeros_(param)

    def gate_forward(self, x):
        # Sparse MoE paper [3] equations 3 & 4
        # (B, E) where E is number of experts
        gate_out = self.g_model(x)
        softplus = nn.functional.softplus(self.g_noise(x)) # this is used as std, which is learnable
        # reparametrization trick used in VAE: N(0, I) * std
        reparam_trick = torch.randn(gate_out.shape, device = x.device) * softplus
        stochastic_gate_out = gate_out + reparam_trick
        # get kth largest, anything below it should be set to -inf
        # Sparse MoE paper [3] equation 5
        k_largest, _ = torch.kthvalue(-stochastic_gate_out, self.top_k, dim=1)
        k_largest = -k_largest.unsqueeze(-1) # (B, 1)
        # needed to implement kth_excluding
        k_next_largest, _ = torch.kthvalue(-stochastic_gate_out, self.top_k+1, dim=1)
        k_next_largest = -k_next_largest.unsqueeze(-1)
        stochastic_gate_out[stochastic_gate_out < k_largest] = -float("inf")
        # (B, E)
        stochastic_gate_out = torch.nn.functional.softmax(stochastic_gate_out, dim=1)

        # Sparse MoE paper [3] equation 9
        # For each expert, use k_next_largest as threshold if that expert is the kth (marginal) expert
        # Otherwise use k_largest
        # (B, E) - True where expert would be exactly kth
        is_kth_expert = (stochastic_gate_out == k_largest) & (stochastic_gate_out != -float("inf"))
        # (B, E) - threshold excluding each expert
        threshold_excluding = torch.where(is_kth_expert, k_next_largest, k_largest)
        
        # probability that, given the expected (mean) + noise (std) of the output from the gate for each sample on each expert, we will get the score that is > threshold
        # z = (threshold - μ) / σ, where z score converts the score of N(μ, σ^2) to the score of N(0, I)
        # P(Y <= threshold) = Φ(z)
        # P(Y > T) = 1 - Φ(z) = Φ(-z) = (μ - threshold) / σ
        # -z = (μ - threshold) / σ
        z_score = (gate_out - threshold_excluding) / (softplus + 1e-9)
        # use precomputed cdf of standard normal
        e_prob = self.std_normal.cdf(z_score)

        return stochastic_gate_out, e_prob
    
    def forward(self, x):
        x_0 = x
        x_l = x

        # Reset importance and load for this forward pass
        if self.num_experts > 1:
            self.importance = torch.zeros(self.num_experts, device=x.device)
            self.load = torch.zeros(self.num_experts, device=x.device)

        # must calculate gate output, then pick top-k experts and calculate weighted sum
        # is there a way to do it in parallel instead of for loop?
                
        if not self.low_rank:
            # DCNv1: x𝑙+1 = x0 * (x𝑙^T @ w𝑙) + b𝑙 + x𝑙
            # OR
            # DCNv2: x𝑙+1 = x0 ⊙ (𝑊𝑙x𝑙 + b𝑙) + x𝑙
            for l in range(self.layers):
                if self.num_experts > 1:
                    gate_out, e_prob = self.gate_forward(x_l)
                    # Sparse MoE paper [3] equation 6
                    # (E, )
                    self.importance += torch.sum(gate_out, dim=0) # batchwise and layerwise sum for each expert
                    # Sparse MoE paper [3] equation 10
                    # (E, )
                    self.load += torch.sum(e_prob, dim = 0) # batchwise and layerwise sum for each expert

                    # weighted sum of experts, weights based on the gate output
                    # Initialize output - loop through each expert and add their weighted contribution to the expert_output
                    expert_outputs = torch.zeros_like(x_l)
                    for expert_id in range(self.num_experts):
                        # Get samples that use this expert
                        # returns a 1D vector of indices
                        expert_batch = gate_out[:,expert_id].nonzero(as_tuple=True)[0]

                        # skip an expert if no samples were chosen for it. This is the real benefit of sparse MoE.
                        # High capacity with 1000s of experts, but only very few are chosen during forward pass.
                        if len(expert_batch) == 0:
                            # skip a loop
                            continue
                        
                        # DCNv2 Paper [2] equation 3
                        # (_B,) where _B is a subsample of B
                        G_i = gate_out[expert_batch, expert_id].unsqueeze(-1)  # (_B, 1) for broadcasting

                        # expert forward pass
                        # (_B, D)
                        E_i = x_0[expert_batch] * self.e_models[l][expert_id](x_l[expert_batch])
                        # Accumulate weighted expert outputs
                        expert_outputs[expert_batch] += G_i * E_i
                    
                    # Add residual connection ONCE after all experts
                    x_l = expert_outputs + x_l
                else:
                    x_l = x_0 * self.e_models[l][0](x_l) + x_l
        else:
            # DCNv2 low-rank: x𝑙+1 = x0 ⊙ (𝑈𝑙(𝑉𝑙^T @ x𝑙) + b𝑙) + x𝑙
            for l in range(self.layers):
                if self.num_experts > 1:
                    gate_out, e_prob = self.gate_forward(x_l)
                    # Sparse MoE paper [3] equation 6
                    # (E, )
                    self.importance += torch.sum(gate_out, dim=0) # batchwise and layerwise sum for each expert
                    # Sparse MoE paper [3] equation 10
                    # (E, )
                    self.load += torch.sum(e_prob, dim = 0) # batchwise and layerwise sum for each expert

                    # weighted sum of experts, weights based on the gate output
                    # Initialize output - loop through each expert and add their weighted contribution to the expert_output
                    expert_outputs = torch.zeros_like(x_l)
                    for expert_id in range(self.num_experts):
                        # Get samples that use this expert
                        # returns a 1D vector of indices
                        expert_batch = gate_out[:,expert_id].nonzero(as_tuple=True)[0]
                        if len(expert_batch) == 0:
                            # skip a loop
                            continue
                        
                        # DCNv2 Paper [2] equation 3
                        # (_B,) where _B is a subsample of B
                        G_i = gate_out[expert_batch, expert_id].unsqueeze(-1)  # (_B, 1) for broadcasting

                        # expert forward pass
                        # x𝑙+1 = ∑︁𝐺𝑖(x𝑙)𝐸𝑖(x𝑙)+x𝑙
                        # 𝐸𝑖(x𝑙)= x0 ⊙ (𝑈𝑖𝑙(𝑉𝑖𝑙^T @ x𝑙) +b𝑙)
                        # layer l, expert i
                        U_l_i = self.e_models[l][expert_id]["U"]
                        V_l_i = self.e_models[l][expert_id]["V"]
                        b_l_i = self.e_models[l][expert_id]["bias"]
                        # (_B, D) @ (D, r) @ (D, r)^T -> (_B, D)
                        E_i = x_0[expert_batch] * (torch.matmul(torch.matmul(x_l[expert_batch], V_l_i), U_l_i.T) + b_l_i)
                        # Accumulate weighted expert outputs
                        expert_outputs[expert_batch] += G_i * E_i
                    # Add residual connection ONCE after all experts
                    x_l = expert_outputs + x_l

                else:
                    U_l = self.e_models[l][0]["U"]
                    V_l = self.e_models[l][0]["V"]
                    b_l = self.e_models[l][0]["bias"]
                    # slighly modified to work with batch dimension
                    # (B, D) @ (D, r) @ (D, r)^T -> (B, D)
                    x_l = x_0 * (torch.matmul(torch.matmul(x_l, V_l), U_l.T) + b_l) + x_l 
        return x_l
    
class DCN(nn.Module):
    def __init__(self, 
                 dense_size = 1, 
                 emb_layers = [], 
                 dnn_layers = [768, 768], 
                 cross_layers = 4, 
                 low_rank = False, 
                 r = 256, 
                 v1 = False, 
                 gate_layers = [], 
                 num_experts = 1,
                 top_k = 1,
                 qr_flag = False, 
                 stacked = False,
                ):
        """
        Args:
            dense_size (Integer): number of dense features.
        """
        super().__init__()
        self.input_size = dense_size+sum([x[1] for x in emb_layers])
        self.emb_layers = emb_layers
        self.dnn_layers = dnn_layers
        self.stacked = stacked
        # cross params
        self.cross_layers = cross_layers
        self.low_rank = low_rank
        self.r = r
        self.v1 = v1
        self.gate_layers = gate_layers
        self.num_experts = num_experts
        self.top_k = top_k
        

        # initialize embedding tables
        self.emb_l = nn.ModuleList()        
        for shapes in emb_layers:
            num_emb, emb_d = shapes
            if qr_flag:
                self.emb_l.append(QREmbeddingBag(num_categories=num_emb, embedding_dim=emb_d, num_collisions=int(math.sqrt(num_emb))))
            else:
                self.emb_l.append(nn.EmbeddingBag(num_embeddings=num_emb, embedding_dim=emb_d))

        # initialize DNN layers
        dnn_steps = nn.ModuleList()
        for i in range(len(dnn_layers)):
            if i == 0:
                dnn_steps.append(nn.Linear(in_features=self.input_size, out_features=dnn_layers[i]))
            else:
                dnn_steps.append(nn.Linear(in_features=dnn_layers[i-1], out_features=dnn_layers[i]))
            dnn_steps.append(nn.ReLU())
        self.dnn = nn.Sequential(*dnn_steps)

        # initialize CrossNet layers
        self.cross = CrossNet(input_size = self.input_size, 
                              layers = cross_layers, 
                              low_rank = low_rank, 
                              r = r, 
                              v1 = v1,
                              gate_layers = gate_layers,
                              num_experts = num_experts,
                              top_k = top_k)

        # initialize projection layer
        self.projection = nn.Linear(in_features= dnn_layers[-1] if self.stacked else self.input_size+dnn_layers[-1], out_features=1)

        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.EmbeddingBag)):
            torch.nn.init.xavier_normal_(module.weight)

    def forward(self, x, emb_indices, emb_offsets):
        """
        Args:
            x (torch.Tensor): dense input features.
            emb_indices (torch.Tensor): embedding indices for k tables and B batch size of shape (k, B).
            emb_offsets (torch.Tensor): embedding offsets for k tables and B batch size of shape (k, B).
        Returns:
            Tensor output of shape (B, 1).
        """

        # embedding lookup across all sparse features
        emb_out = []
        K = emb_indices.size(0)
        for k in range(K):
            emb_out.append(self.emb_l[k](emb_indices[k], emb_offsets[k]))
        # (B, dense) + (B, sparse) -> (B, H) where H = dense + sparse
        T = torch.cat([x]+emb_out, dim=1)
        # (B, H) -> (B, H)
        f_cross = self.cross(T)
        if self.stacked:
            # (B, H) -> (B, D) where D = last dnn hidden size
            out = self.dnn(f_cross)
        else:
            # parallel
            # (B, H) -> (B, D)
            f_dnn = self.dnn(T)
            # (B, H) + (B, D) -> (B, H + D)
            out = torch.cat([f_cross, f_dnn], dim=1)

        # (B, D) -> (B, 1) if stacked
        # (B, H + D) -> (B, 1) if parallel
        return self.projection(out)


class Solver:
    def __init__(self, 
                 model, 
                 data,
                 optimizer,
                 device, 
                 epochs, 
                 checkpoint_dir = "./checkpoints", 
                 checkpoint_every = 1, 
                 verbose = True, 
                 print_every = 1, 
                 reset = False,
                 importance_weight = 0,
                 load_weight = 0
                ):
        self.model = model
        self.data = data
        self.device = device
        self.epochs = epochs
        self.verbose = verbose
        self.reset = reset
        self.print_every = print_every
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every  
        self.optimizer = optimizer
        self.importance_weight = importance_weight
        self.load_weight = load_weight
        
        self.loss_history = []
        self.loss_history_batch = []
        self.load_cv_history_batch = []
        self.importance_cv_history_batch = []

        self.model.to(device)
    
    def _step(self):

        total_loss = 0
        nbatches = len(self.data)
        # counter = 0
        for x, emb_offsets, emb_indices, target in self.data:
            # print(f"[{counter}/{nbatches}]")
            # counter += 1
            
            x = x.to(self.device)
            emb_indices = emb_indices.to(self.device)
            emb_offsets = emb_offsets.to(self.device)
            target = target.to(self.device).float()
            if target.dim() == 1:
                target = target.unsqueeze(1)


            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(x, emb_indices, emb_offsets)
            loss = nn.functional.binary_cross_entropy_with_logits(
                input=logits,
                target=target,
                reduction="mean",
            )
            if self.model.num_experts > 1:
                # add load and importance losses
                # Sparse MoE paper [3] equation 7
                importance_std, importance_mean = torch.std_mean(self.model.cross.importance)
                importance_cv = importance_std / (importance_mean + 1e-9)
                importance_loss = self.importance_weight * importance_cv**2
                # Sparse MoE paper [3] equation 11
                load_std, load_mean = torch.std_mean(self.model.cross.load)
                load_cv = load_std / (load_mean + 1e-9)
                load_loss = self.load_weight * load_cv**2

                loss = loss + importance_loss + load_loss

                # overtime the loss function should force the variance for importance and load to decrease, so CV would decrease
                self.importance_cv_history_batch.append(float(importance_cv.item()))
                self.load_cv_history_batch.append(float(load_cv.item()))




            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            self.loss_history_batch.append(float(loss.item()))

        return total_loss / nbatches

    
    def _save(self, loss, epoch, filepath):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint = {
            "epoch" : epoch,
            "loss" : loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at {filepath}")

    def _load(self, filepath):
        if not os.path.exists(filepath):
            return None, None
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        return epoch, loss
    
    def train(self):
        start_epoch = 0
        if not self.reset:
            saved_epoch, saved_loss = self._load(os.path.join(self.checkpoint_dir, "last_checkpoint.pth"))

            # if loading a saved epoch, then continue from the last epoch
            if saved_epoch is not None:
                self.loss_history.append(saved_loss)
                print(f"Load [{saved_epoch}/{self.epochs}] Loss {self.loss_history[-1]:.6f}")
                start_epoch = saved_epoch + 1

        # Set the model to training mode
        self.model.train()

        for epoch in range(start_epoch, self.epochs):
            epoch_start_time = time.time()
            epoch_loss = self._step()
            epoch_end_time = time.time()
            
            if self.verbose and epoch % self.print_every == 0:
                print(f"[{epoch}/{self.epochs}] Loss: {epoch_loss:.6f} time: {(epoch_end_time - epoch_start_time) / 60.0}m")
            if epoch % self.checkpoint_every == 0:
                self._save(loss = epoch_loss, epoch = epoch, filepath = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
                self._save(loss = epoch_loss, epoch = epoch, filepath = os.path.join(self.checkpoint_dir, f"last_checkpoint.pth"))
            
            self.loss_history.append(epoch_loss)