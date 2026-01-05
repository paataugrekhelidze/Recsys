import torch.nn as nn
import torch
from tricks.qr_embedding_bag import QREmbeddingBag
import math
import os
import time

class DLRM(nn.Module):
    def __init__(self, input_size, b_mlp_layers, t_mlp_layers, emb_layers, qr_flag = True):
        """
        Deep Learning Recommendation Model implementation.

        Args:
            input_size (Int)
            b_mlp_layers (List): List of hidden layers for the bottom mlp architecture.
            t_mlp_layers (List): List of hidden layers for the top mlp architecture.
            emb_layers (List): List of K tuples, each specifying the embedding table dimensions.
        """
        super().__init__()
        b_layers = nn.ModuleList()
        t_layers = nn.ModuleList()

        # initialize bottom mlp
        for i in range(len(b_mlp_layers)):
            if i == 0:
                b_layers.append(nn.Linear(in_features=input_size, out_features=b_mlp_layers[i]))
            else:
                b_layers.append(nn.Linear(in_features=b_mlp_layers[i-1], out_features=b_mlp_layers[i]))
            b_layers.append(nn.ReLU())
        
        # flattened upper triangular of the interaction matrix
        # 1+K + 2N = (1+K)^2
        # N = (1+K)*(K) // 2

        # t_mlp_layers = dense features from botton_mlp + interactions
        K = len(emb_layers)
        t_mlp_input_size = b_mlp_layers[-1] + (1+K)*(K) // 2

        # initialize top mlp
        for i in range(len(t_mlp_layers)):
            if i == 0:
                t_layers.append(nn.Linear(in_features=t_mlp_input_size, out_features=t_mlp_layers[i]))
            else:
                t_layers.append(nn.Linear(in_features=t_mlp_layers[i-1], out_features=t_mlp_layers[i]))
            if i != len(t_mlp_layers)-1:
                t_layers.append(nn.ReLU())

        self.b_mlp = nn.Sequential(*b_layers)
        self.t_mlp = nn.Sequential(*t_layers)

        # initialize embedding tables
        self.emb_l = nn.ModuleList()        
        for i, shapes in enumerate(emb_layers):
            num_emb, emb_d = shapes
            if qr_flag:
                self.emb_l.append(QREmbeddingBag(num_categories=num_emb, embedding_dim=emb_d, num_collisions=int(math.sqrt(num_emb))))
            else:
                self.emb_l.append(nn.EmbeddingBag(num_embeddings=num_emb, embedding_dim=emb_d))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
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
        
        # (B, (1+K)*D) -> (B, 1+K, D)
        batch_size, D = dense.shape
        T = torch.cat([dense] + sparse, dim=1).view((batch_size, -1, D))
        # print(f"T: {T.shape}")

        # (B, 1+K, D) x (B, D, 1+K) -> (B, 1+K, 1+K)
        Z = torch.bmm(T, torch.transpose(T, 1, 2))

        # print(f"Z: {Z.shape}")

        # get upper triangular for unique interactions, exlude diagonal
        row, col = torch.triu_indices(Z.shape[1], Z.shape[2], offset=1)
        # (B, 1+K, 1+K) -> (B, N) where N = (1+K)*(K) // 2
        Z_flat = Z[:, row, col]
        # print(f"Z_flat: {Z_flat.shape}")

        # combine original dense featues and flattened upper triangular of interactions
        # (B, N+D)
        combined = torch.cat([dense] + [Z_flat], dim=1)

        return combined
        

    def forward(self, x, emb_indices, emb_offsets):
        """
        Args:
            x (torch.Tensor): dense input features.
            emb_indices (torch.Tensor): embedding indices for k tables and B batch size of shape (k, B).
            emb_offsets (torch.Tensor): embedding offsets for k tables and B batch size of shape (k, B).
        Returns:
            Tensor output of shape (B, 1).
        """
        
        # step 1: score bottom MLP for dense features
        # (B, input) -> (B, D)
        b_mlp_out = self.b_mlp(x)

        # step 2: embedding lookup across all sparse features
        emb_out = []
        K = emb_indices.size(0)
        for k in range(K):
            emb_out.append(self.emb_l[k](emb_indices[k], emb_offsets[k]))
        
        # print(b_mlp_out.shape, len(emb_out))

        # step 3: calulate interaction matrix
        z = self._interact(b_mlp_out, emb_out)
        # print(z.shape)

        # step 4: score top MLP using output from the interaction op
        t_mlp_out = self.t_mlp(z)

        return t_mlp_out

class DLRMSolver:
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