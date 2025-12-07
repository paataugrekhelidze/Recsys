import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import wandb

class UserMovieDataset(Dataset):
    def __init__(self, datapath, moviepath, holdout_split=0.8, device = "cpu"):
        # Load the dataset
        self.filepath = datapath
        self.device = device
        self.data = pd.read_csv(self.filepath)
        self.movies = pd.read_csv(moviepath)

        
        # Preload unique movieIds to define the columns
        self.movie_ids = sorted(self.movies['movieId'].unique())
        self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        
        # Preload unique userIds
        self.user_ids = self.data['userId'].unique()

        # Flag to indicate if this dataset is for holdout split
        self.holdout_split = holdout_split

    def __len__(self):
        # Number of unique users
        return len(self.user_ids)
    
    def vector_to_df(self, data = None, R = 100):
        if data is None:
            return None
        # tuple of values and original indicies
        data = torch.sort(data, dim=1, descending=True)
        # use a special index, a rare movie index, to imply nonclicked positions
        no_click_ix = -5
        data[1][data[0] == 0] = no_click_ix
        # keep only top R clicks
        indicies = data[1][:, :R]
        # map tensor idx to the actual movieId
        indicies.apply_(lambda x: self.movie_ids[x])

        # create pandas dataframe of userId and associated top R movieIds
        df = [{"userId": self.user_ids[i].item(), **{f"movie {j}" : indicies[i][j].item() for j in range(indicies.size(1))}} for i in range(indicies.size(0))]
        df = pd.DataFrame(df)

        # map movieId to associated title
        movie_map_dict = self.movies.set_index('movieId')['title'].to_dict()

        # Handle the special case
        movie_map_dict[self.movie_ids[no_click_ix].item()] = "None"

        # cast int cols to object to avoid warnings
        df[df.columns[1:]] = df[df.columns[1:]].astype(object)

        # Apply the dictionary map to the DataFrame slice
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda col: col.map(movie_map_dict))
        
        return df, data[0][:, :R]
        

    def __getitem__(self, idx):
        # Get the userId for the given index
        user_id = self.user_ids[idx]
        
        # Filter the data for the current user
        user_data = self.data[self.data['userId'] == user_id].sort_values(by='timestamp')
        
        # Create a sparse vector for the user's interactions
        user_vector = torch.zeros(len(self.movie_ids))
        for _, row in user_data.iterrows():
            movie_idx = self.movie_id_to_index[row['movieId']]
            user_vector[movie_idx] = row['click']

        if self.holdout_split is not None:
            # 80% (default) for inference, 20% (default) for evaluation
            split_idx = int(len(user_data) * self.holdout_split)
            inference_data = user_data.iloc[:split_idx]
            test_data = user_data.iloc[split_idx:]

            # Inference vector
            X = torch.zeros(len(self.movie_ids))
            for _, row in inference_data.iterrows():
                movie_idx = self.movie_id_to_index[row['movieId']]
                X[movie_idx] = row['click']
            
            # Test vector
            Y = torch.zeros(len(self.movie_ids))
            for _, row in test_data.iterrows():
                movie_idx = self.movie_id_to_index[row['movieId']]
                Y[movie_idx] = row['click']
            
            return user_id, X, Y
        
        return user_id, user_vector


# Encoder
class VAEEncoder(nn.Module):
    """
    Encodes input vector x into a distribution of a latent factor z.
    The Enoder assumes the distribution to be Gaussian and learns to predict mean and variance.
    (Netflix Paper section 4.3)
    Identity (x) -> 600 -> mean, var (2*K)
    """
    def __init__(self, identity_size, hidden_size, K, dropout_p = 0.5):
        super().__init__()

        self.dropout = nn.Dropout(p = dropout_p)
        self.hidden_layer = nn.Linear(identity_size, hidden_size)
        self.latent_layer = nn.Linear(hidden_size, 2*K)
        # initialize weights for linear layers
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            # nn.init.zeros_(module.bias)
            nn.init.trunc_normal_(module.bias, mean=0.0, std=0.001, a=-0.002, b=0.002)
    
    # returns mean, variance
    def forward(self, x):
        """
        Encoder forward pass function. takes user history vector x and output mean and variance. first K elements predict mean for each
        latent dimension, and other K elements predict variance.

        Inputs:
         - x: user-movie click history, of shape
              (N, D), where N is the batch size, D is the feature size
        Returns:
         - out: Guassian distribution (mean_i, variance_i + K) for each dimension i (N, 2*K)
        """
        # l2 normalize, divide vector by its l2 magnitude
        # l2 magnitude = sqrt(x_0^2 + x_1^2 + .....)
        x = F.normalize(x, p=2, dim=1) # same as x / torch.norm(x, p=2, dim=1)
        x = self.dropout(x)
        x = F.tanh(self.hidden_layer(x))
        out = self.latent_layer(x)
        return out


# Decoder
class VAEDecoder(nn.Module):
    """
    Reconstructs vector x from a latent factor z.
    (Netflix Paper section 4.3)
    z (K) -> 600 -> Identity (x)
    """
    def __init__(self, identity_size, hidden_size, K):
        super().__init__()

        self.hidden_layer = nn.Linear(K, hidden_size)
        self.output_layer = nn.Linear(hidden_size, identity_size)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            # nn.init.zeros_(module.bias)
            nn.init.trunc_normal_(module.bias, mean=0.0, std=0.001, a=-0.002, b=0.002)
    
    # returns mean, variance
    def forward(self, z):
        """
        Decoder forward pass function. Takes latent vector z and outputs vector x. 

        Inputs:
         - z: user-movie click history, of shape
              (N, K), where N is the batch size, K is the latent vector size
        Returns:
         - out: user-movie click history, of shape
              (N, D), where N is the batch size, D is the feature size
        """

        z = F.tanh(self.hidden_layer(z))
        out = self.output_layer(z)
        return out

# VAE Model
class VAEModel(nn.Module):
    """
    Combines VAE Encoder and Decoder layers to comput forward pass for Train and Inference.

    Train forward:  Identity (x) -> 600 -> mean, var (sample z) -> 600 -> Identity (x)
    Inference forward: Identity (x) -> 600 -> mean (z) -> 600 -> Identity (x)

    Only sample z during training to allow the encoder to learn how to compress input data x into
    a Gaussian distribution. Only use mean to infer z (Netflix Paper section 2.4) during inference so the inference forward is consistent
    The Netflix VAE paper states (Section 2.2.1, superscript 4) to predict log(variance) in the encoder layer, from which 
    we recover the variance.
    """
    def __init__(self, identity_size, hidden_size = 600, K = 200, dropout_p = 0.5):

        super().__init__()
        self.identity_size = identity_size
        self.hidden_size = hidden_size
        self.K = K
        self.dropout_p = dropout_p

        self.encoder = VAEEncoder(identity_size = self.identity_size, hidden_size = self.hidden_size, K = self.K, dropout_p = self.dropout_p)
        self.decoder = VAEDecoder(identity_size = self.identity_size, hidden_size = self.hidden_size, K = self.K)


    def forward(self, x):
        """
        Encodes input vector x into a distribution of a latent factor z. Either samples or uses mean to get vector z,
        then decodes latent vector z back to x.

        Inputs:
         - x: user-movie click history, of shape
              (N, D), where N is the batch size, D is the feature size
        Returns:
         - out: predicted reconstruction of user-movie click history, of shape
              (N, D), where N is the batch size, D is the feature size
        """

        x = self.encoder(x) # (N, 2*K)

        mu = x[:, :self.K]
        log_var = x[:, self.K:]
        
        z = mu # (N, K)
        if self.training:
            # if we treat output of the encoder as log(var) then we need to convert it to var = exp(log(var))
            var = log_var.exp()
            std = torch.sqrt(var)
            # reparametrization trick, z = mu + std * epsilon
            # section 2.2.1 Algorithm 1
            epsilon = torch.randn(*std.shape, device = x.device) # N(0, I)
            z = mu + torch.mul(std, epsilon)
        
        out = self.decoder(z) # (N, D)
        if self.training:
            # return output both from the encoder and decoder, needed to calculate loss terms
            return mu, log_var, out
        return out
    
class VAESolver:
    def __init__(self, 
                 model, 
                 data, 
                 device, 
                 epochs, 
                 batch_size,
                 lr = 0.001,
                 num_workers = 0, 
                 checkpoint_dir = "./checkpoints", 
                 checkpoint_every = 5, 
                 verbose = True, 
                 print_every = 5, 
                 reset = False, 
                 wandb_api = None,
                 anneal_steps = 20000,
                 anneal_cap = 0.2,
                 lam = 0
                ):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.verbose = verbose
        self.reset = reset
        self.print_every = print_every
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.wandb_api = wandb_api
        self.anneal_steps = anneal_steps
        self.anneal_cap = anneal_cap
        self.lam = lam
        # sysctl -n hw.ncpu -> 12 cores on my laptop
        self.data = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay=self.lam) # to be modified, lr, maybe weight_decay
        # paper mentions that beta reaches 1 at epoch = 80 (section 2.2.2 Figure 1)
        # use this info to calculate rate of annealing
        self.beta = 0
        self.update_count = 0
        self.loss_history = []

        self.wandb_run = None
        # wandb login
        if self.wandb_api:
            os.environ['WANDB_API_KEY'] = self.wandb_api
            wandb.login()

            # Start a new wandb run to track this script.
            self.wandb_run = wandb.init(
                # Set the wandb project where this run will be logged.
                project="Netflix-VAE",
                # Track hyperparameters and run metadata.
                config={
                    "learning_rate": self.lr,
                    "Optimizer": "Adam",
                    "architecture": "VAE",
                    "dataset": "MovieLens20M",
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "Max_Beta": self.anneal_cap,
                    "Anneal_steps": self.anneal_steps,
                    "l2_lambda": self.lam
                },
            )

            # Define 'epoch' as the default x-axis for all metrics
            wandb.define_metric("epoch", step_metric = "epoch")
    
    def _step(self):
        total_loss = 0
        total_recon = 0
        total_kld = 0
        total_samples = 0
        # batch_i = 0
        for _, x_train_batch in self.data:
            x_train_batch = x_train_batch.to(self.device)

            mu, log_var, out = self.model(x_train_batch)

            # stop increasing beta at 0.2 (section 2.2.2)
            self.beta = min(self.beta + (1.0 * self.update_count / self.anneal_steps), self.anneal_cap)

            # multinomial log likelihood + KL Divergence (section 2.2.2 equation 6)
            loss, recon, kld = vae_loss_function(recon_out=out, x_target=x_train_batch, mu=mu, log_var=log_var, beta=self.beta)
            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            total_samples += x_train_batch.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print(f"[{batch_i}/{len(self.data)}]")
            # batch_i += 1
            self.update_count += 1
            
        return total_loss, total_recon, total_kld, total_samples, len(self.data)
    
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
            total_loss, total_recon, total_kld, n_samples, n_batches = self._step()
            epoch_end_time = time.time()

            loss_p_batch = total_loss / n_batches
            recon_p_batch = total_recon / n_batches
            kld_p_batch = total_kld / n_batches
            loss_p_sample = total_loss / n_samples

            
            if self.verbose and epoch % self.print_every == 0:
                print(f"[{epoch}/{self.epochs}] Loss per batch: {loss_p_batch:.6f} Loss per sample: {loss_p_sample:.6f} Recon_loss pre batch: {recon_p_batch:.6f} beta_KLD per batch: {kld_p_batch:.6f} time:  {(epoch_end_time - epoch_start_time) / 60.0}m")
            if epoch % self.checkpoint_every == 0:
                self._save(loss = loss_p_batch, epoch = epoch, filepath = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
                self._save(loss = loss_p_batch, epoch = epoch, filepath = os.path.join(self.checkpoint_dir, f"last_checkpoint.pth"))
            
            self.loss_history.append(loss_p_batch)
            if self.wandb_run:
                # self.wandb_run.log({"Recall@20": recall, "NDCG@20": ndcg, "Loss": loss, "epoch": epoch})
                self.wandb_run.log({"loss_p_batch": loss_p_batch, "loss_p_sample": loss_p_sample, "epoch": epoch})
        if self.wandb_api:
            wandb.finish()



def recall_at_r(predicted, actual, R):
    """
    Sum of Recall@R for a batch of users.

    Args:
        predicted (torch.Tensor): Predicted probs of user-movie interactions (N, D).
        actual (torch.Tensor): Ground truth user-movie interactions (N, D).
        R (int): Number of top predictions to consider.

    Returns:
        Tuple(float, float): Sum Recall@R for the batch, size of the batch
    """

    top_R_results = torch.topk(predicted, k=R, dim=1) # (N, R)
    # capture actual values of the top predicted indices
    actual_gather = torch.gather(actual, dim=1, index=top_R_results.indices) # (N, R)
    recall = torch.sum(actual_gather, dim=1, keepdim=True) / (torch.sum(actual, dim=1, keepdim=True) + 1e-9) #(N, 1)
        
    return recall.sum().item(), recall.size(0)

def ndcg_at_r(predicted, actual, R, device):
    """
    Sum of NDCG@20 for a batch of users.
    
    Args:
        predicted (torch.Tensor): Predicted probs of user-movie interactions (N, D).
        actual (torch.Tensor): Ground truth user-movie interactions (N, D).
        R (int): Number of top predictions to consider.

    Returns:
        Tuple(float, float): Sum NDCG@R for the batch, size of the batch
    """

    # calculate ideal DCG
    # get top indices, divide by log(i+1) where i is the rank/position of an element
    actual_sorted = torch.sort(actual, dim=-1, descending=True)
    discount_factor = torch.pow(torch.log2(torch.arange(1, R + 1, dtype=actual.dtype, device = device) + 1), exponent=-1)
    idcg = actual_sorted[0][:,:R] * discount_factor # (N, R)
    idcg = torch.sum(idcg, dim=1, keepdim=True)
    
    # calculate DCG from predicted values
    predicted_sorted = torch.sort(predicted, dim=-1, descending=True)
    # actual relevance values (here its 1 or 0) for top indices predicted by model
    predicted_relevance = torch.gather(actual, dim=1, index=predicted_sorted[1][:,:R])
    predicted_dcg = predicted_relevance * discount_factor # (N, R)
    predicted_dcg = torch.sum(predicted_dcg, dim=1, keepdim=True) # (N, 1)

    # calculate NDCG
    ndcg = predicted_dcg / (idcg + 1e-9) # (N, 1)
    return ndcg.sum().item(), ndcg.size(0)

def vae_loss_function(recon_out, x_target, mu, log_var, beta=1.0):
    # Reconstruction Loss (Negative Log-Likelihood of Multinomial)
    recon = F.cross_entropy(recon_out, x_target, reduction='sum')

    # KL Divergence
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2) I asked Gemini for this one hehe!
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Negative ELBO (Loss = recon + kld)
    # Note: We apply the annealing factor beta to the KL term.
    loss = recon + beta * kld
    return loss, recon, beta * kld