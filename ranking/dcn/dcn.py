import torch.nn as nn
import torch
import math
from tricks.qr_embedding_bag import QREmbeddingBag
import os
import time

class CrossNet(nn.Module):
    def __init__(self, input_size, layers = 4, low_rank = False, r = 256, v1 = False):
        super().__init__()
        self.input_size = input_size
        self.layers = layers
        self.low_rank = low_rank
        self.r = r
        self.v1 = v1

        steps = nn.ModuleList()
        if not low_rank:
            if not v1:
                # x𝑙+1 = x0 ⊙ (𝑊𝑙x𝑙 +b𝑙)+x𝑙 - initialize 𝑊𝑙 and b𝑙 - DCNv2 Paper [2] equation 1
                # full matrix
                for _ in range(layers):
                    steps.append(nn.Linear(in_features=input_size, out_features=input_size, bias=True))
            else:
                # x𝑙+1 = x0 * (x𝑙^T @ w𝑙) + b𝑙 + x𝑙 - initialize w𝑙 and b𝑙 - DCNv1 Paper [1] equation 3
                # rank 1 vector
                for _ in range(layers):
                    steps.append(nn.Linear(in_features=input_size, out_features=1, bias=True))
            self.cross_net = nn.Sequential(*steps)
        else:
            # x𝑙+1 = x0 ⊙ (𝑈𝑙(𝑉𝑙^T @ x𝑙) + b𝑙) + x𝑙 - initialize 𝑈𝑙, 𝑉𝑙, and b𝑙 - DCNv2 Paper [2] equation 2
            # rank_r matrix
            rank = max(1, r)
            weight_shape = (input_size, rank)
            bias_shape = (input_size,)
            for _ in range(layers):
                U = nn.Parameter(torch.empty(weight_shape))
                V = nn.Parameter(torch.empty(weight_shape))
                b = nn.Parameter(torch.empty(bias_shape))
                steps.append(nn.ParameterDict({
                    "U": U,
                    "V": V,
                    "bias": b
                }))
            self.cross_net = steps

        self.apply(self._init_weights)    

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

    
    def forward(self, x):
        x_0 = x
        x_l = x
        
        if not self.low_rank:
            # DCNv1: x𝑙+1 = x0 * (x𝑙^T @ w𝑙) + b𝑙 + x𝑙
            # OR
            # DCNv2: x𝑙+1 = x0 ⊙ (𝑊𝑙x𝑙 + b𝑙) + x𝑙
            for layer in self.cross_net:
                x_l = x_0 * layer(x_l) + x_l
        else:
            # DCNv2 low-rank: x𝑙+1 = x0 ⊙ (𝑈𝑙(𝑉𝑙^T @ x𝑙) + b𝑙) + x𝑙
            for l in range(self.layers):
                U_l = self.cross_net[l]["U"]
                V_l = self.cross_net[l]["V"]
                b_l = self.cross_net[l]["bias"]
                # slighly modified to work with batch dimension
                # (b, D) @ (D, r) @ (D, r)^T -> (b, D)
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
                 qr_flag = False, 
                 stacked = False
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
        self.cross = CrossNet(input_size = self.input_size, layers = cross_layers, low_rank = low_rank, r = r, v1 = v1)

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
                 reset = False
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
        
        self.loss_history = []
        self.loss_history_batch = []

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