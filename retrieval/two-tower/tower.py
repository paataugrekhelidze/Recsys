from functools import lru_cache
import torch
import datasets
from collections import defaultdict
from typing import Dict, List, Literal
from datasets import Dataset, DatasetDict, load_dataset
import os
import time
import numpy as np

class YambdaDataset:
    INTERACTIONS = frozenset([
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ])

    def __init__(
        self,
        dataset_type: Literal["flat", "sequential"] = "flat",
        dataset_size: Literal["50m", "500m", "5b"] = "50m"
    ):
        assert dataset_type in {"flat", "sequential"}
        assert dataset_size in {"50m", "500m", "5b"}
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

    def interaction(self, event_type: Literal[
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ]) -> Dataset:
        assert event_type in YambdaDataset.INTERACTIONS
        return self._download(f"{self.dataset_type}/{self.dataset_size}", event_type)

    def audio_embeddings(self) -> Dataset:
        return self._download("", "embeddings")

    def album_item_mapping(self) -> Dataset:
        return self._download("", "album_item_mapping")

    def artist_item_mapping(self) -> Dataset:
        return self._download("", "artist_item_mapping")

    @staticmethod
    def _download(data_dir: str, file: str) -> Dataset:
        data = load_dataset("yandex/yambda", data_dir=data_dir, data_files=f"{file}.parquet")
        # Returns DatasetDict; extracting the only split
        assert isinstance(data, DatasetDict)
        return data["train"]

class MusicWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data: datasets.arrow_dataset.Dataset, 
                 n: int = 10, 
                 global_t_max: int = None,
                 max_windows_per_user = 100
                ):
        self.user_sequences = data
        self.n = n
        self.global_t_max = global_t_max
        self.UNKNOWN_ID = -1
        self.max_windows_per_user = max_windows_per_user

        self._generate_samples()

    
    def _generate_samples(self):
        print("Generating samples from sequences...")
        sample_list = []
        for user_idx, seq in enumerate(self.user_sequences):
            num_items = len(seq["item_id"])
            if num_items > self.n + 1:
                possible_choices = np.arange(num_items-self.n-1) # number of possible starting indices for windows
                if len(possible_choices) > self.max_windows_per_user:
                    # Consuming a millions of interactions is not efficient on my laptop
                    # instead I randomly sample windows per user
                    choices = np.random.choice(a=possible_choices, size=self.max_windows_per_user, replace=False)
                else:
                    choices = possible_choices
                for start_idx in choices:
                    # contains tuple of all possible (user_id, start index for window of size n) combination
                    sample_list.append((user_idx, start_idx))
        # apparently numpy array is more memory efficient
        self.samples = np.array(sample_list, dtype=np.int32)
        print(f"Generated {len(self.samples)} total samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_idx, start_idx = self.samples[idx]
        seq = self.user_sequences[int(user_idx)]
        i = start_idx

        item_ids = seq['item_id']
        artist_ids = seq['artist_id']
        album_ids = seq['album_id']
        timestamps = seq['timestamp']
        track_lengths = seq['track_length_seconds']
        played_ratios = seq['played_ratio_pct']
        
        window = {
            # query tower
            "user_id": seq["uid"],
            "history_id": item_ids[i : i + self.n],
            "history_track_length": track_lengths[i : i + self.n],
            "history_artist_id": artist_ids[i : i + self.n],
            "history_album_id": album_ids[i : i + self.n],

            "seed_id": item_ids[i + self.n],
            "seed_track_length": track_lengths[i + self.n],
            "seed_artist_id": artist_ids[i + self.n],
            "seed_album_id": album_ids[i + self.n],
            
            # many ways to normalize age - BatchNorm, Quantile-based, Min-max, log+z_score, binning,...
            # batchNorm would not be ideal for online serving of recommendation systems - X
            # min-max, log+z - somewhat normal distribution (efficient for streaming data, log addresses skewness in power law distribution by adjusting units between timeframes e.g 0-1hour vs 1hour-1day, z_score centers at 0 with var 1)
            # quantile-based - produces a nice bell-shaped distribution (wanted to use it but not efficient to calculate across millions of records)
            # binning - another possible option (might test it against quantile-based)
            # normalization happens in forward pass for efficiency
            "age": self.global_t_max - timestamps[i + self.n + 1], # "Example Age" (Youtube paper[1] Section 3.3) uses the timestamp of the candidate item interaction
            # "age": self.normalizer["age"].transform([[self.global_t_max - timestamps[i + self.n + 1]]])[0][0], # "Example Age" (Youtube paper[1] Section 3.3) uses the timestamp of the candidate item interaction
            
            # item tower
            "candidate_id": item_ids[i + self.n + 1],
            "candidate_track_length": track_lengths[i + self.n + 1],
            "candidate_artist_id": artist_ids[i + self.n + 1],
            "candidate_album_id": album_ids[i + self.n + 1],
            
            # loss reward
            "played_ratio_pct": played_ratios[i + self.n + 1] # used as a reward weight for pos samples (candidate) in loss func

        }

        query = {
            "user_id": torch.tensor(window["user_id"], dtype=torch.long),
            "history_id": torch.tensor(window["history_id"], dtype=torch.long),
            "history_track_length": torch.tensor(window["history_track_length"], dtype=torch.float32),
            "history_artist_id": torch.tensor(window["history_artist_id"], dtype=torch.long),
            "history_album_id": torch.tensor(window["history_album_id"], dtype=torch.long),

            "seed_id": torch.tensor(window["seed_id"], dtype=torch.long),
            "seed_track_length": torch.tensor(window["seed_track_length"], dtype=torch.float32),
            "seed_artist_id": torch.tensor(window["seed_artist_id"], dtype=torch.long),
            "seed_album_id": torch.tensor(window["seed_album_id"], dtype=torch.long),

            "age": torch.tensor(window["age"], dtype=torch.float32),
        }
        item = {
            "candidate_id": torch.tensor(window["candidate_id"], dtype=torch.long),
            "candidate_track_length": torch.tensor(window["candidate_track_length"], dtype=torch.float32),
            "candidate_artist_id": torch.tensor(window["candidate_artist_id"], dtype=torch.long),
            "candidate_album_id": torch.tensor(window["candidate_album_id"], dtype=torch.long),
        }
            
        reward = torch.tensor(window["played_ratio_pct"], dtype=torch.float32)
        
        return query, item, reward
    
def create_validation_split(dataset, validation_days=7, n = 10):
    """
    Creates a time-based train/validation split.
    
    Args:
        dataset (Dataset): The complete, sorted sequential dataset.
        validation_days (int): The number of most recent days to use for validation.

    Returns:
        A tuple of (train_dataset, validation_dataset).
    """
    print("Creating time-based train/validation split...")
    
    # Assuming timestamps are in seconds. 1 day = 86400 seconds.
    DAY_IN_SECONDS = 86400
    
    # Find the latest timestamp in the entire dataset
    max_timestamp = 0
    for seq in dataset:
        if len(seq['timestamp']) > 0:
            max_timestamp = max(max_timestamp, max(seq['timestamp']))
    
    # Define the time boundary for the split
    split_timestamp = max_timestamp - (validation_days * DAY_IN_SECONDS)
    print(f"Max timestamp: {max_timestamp}, Split timestamp: {split_timestamp}")

    # Filter the dataset into train and validation sets based on the timestamp of the *interaction*
    def split_sequences(seq):
        timestamps = np.array(seq['timestamp'])
        train_mask = timestamps < split_timestamp
        val_mask = timestamps >= split_timestamp
        
        # Create a new sequence for each split
        train_seq = {k: np.array(v)[train_mask].tolist() for k, v in seq.items() if k != 'uid'}
        val_seq = {k: np.array(v)[val_mask].tolist() for k, v in seq.items() if k != 'uid'}
        train_seq['uid'] = val_seq['uid'] = seq['uid']
        
        return train_seq, val_seq

    train_sequences, val_sequences = [], []
    for seq in dataset:
        train_s, val_s = split_sequences(seq)
        # train on user data, even if the validation data not available
        # for similicity, only get users that have at least n interaction, size of a history
        if len(train_s['item_id']) > n: 
            train_sequences.append(train_s)
        if len(val_s['item_id']) > n:
            val_sequences.append(val_s)

    # Create new Hugging Face Datasets from the lists of dicts
    from datasets import Dataset as HFDataset
    train_ds = HFDataset.from_list(train_sequences)
    val_ds = HFDataset.from_list(val_sequences)

    print(f"Train sequences: {len(train_ds)}, Validation sequences: {len(val_ds)}")

    return train_ds, val_ds

def create_item_data(dataset, artist_map, album_map, offset, qt):
    import polars as pl
    import gc
    from datasets import Dataset as HFDataset

    print(f"unique_items: {len(dataset.unique('item_id'))}")

    items_ldf = pl.from_arrow(dataset.data.table).lazy().select(['item_id', 'track_length_seconds'])
    
    # Also make the mappings lazy
    artist_ldf = pl.from_arrow(artist_map.data.table).lazy()
    album_ldf = pl.from_arrow(album_map.data.table).lazy()

    final_ldf = (
        items_ldf
        # Deduplicate items within the items_ldf itself
        .unique(subset=['item_id'], keep='first')
        # Join with artist and album data
        .join(artist_ldf.unique(subset=['item_id'], keep='first'), on="item_id", how="left")
        .join(album_ldf.unique(subset=['item_id'], keep='first'), on="item_id", how="left")
        # Fill nulls that resulted from the left joins
        .fill_null(-1)
    )

    print("Executing lazy plan (collecting results)...")
    final_df = final_ldf.collect(streaming=True)
    print(f"Collected {len(final_df)} unique items.")
    
    # Clean up original dataset immediately to free memory
    del dataset, artist_map, album_map, items_ldf, artist_ldf, album_ldf, final_ldf
    gc.collect()
    
    # Perform Vectorized Transforms
    print("Applying transforms...")
    
    # Candidate ID
    item_ids_np = final_df['item_id'].to_numpy().reshape(-1, 1)
    candidate_ids = offset["item_id"].transform(item_ids_np).flatten()
    
    # Artist/Album offsets
    artist_ids_np = final_df['artist_id'].to_numpy().reshape(-1, 1)
    candidate_artist_ids = offset["artist_id"].transform(artist_ids_np).flatten()
    
    album_ids_np = final_df['album_id'].to_numpy().reshape(-1, 1)
    candidate_album_ids = offset["album_id"].transform(album_ids_np).flatten()
    
    # Track Length Quantile Transform
    lengths_np = final_df['track_length_seconds'].to_numpy().reshape(-1, 1)
    candidate_track_lengths = qt["track_length"].transform(lengths_np).flatten()
    
    # Build final Result
    final_item_data = pl.DataFrame({
        "candidate_id": candidate_ids,
        "candidate_artist_id": candidate_artist_ids,
        "candidate_album_id": candidate_album_ids,
        "candidate_track_length": candidate_track_lengths
    })
    
    return HFDataset.from_polars(final_item_data)

class QueryTower(torch.nn.Module):
    def __init__(self, input_size: int, 
                 hidden_size: List[int], 
                 user_num_embeddings: int, 
                 user_embed_size: int, 
                 item_embed: torch.nn.Embedding, 
                 artist_embed: torch.nn.Embedding, 
                 album_embed: torch.nn.Embedding,
                 log_age_mean: float,
                 log_age_std: float):
        super().__init__()
        
        # user id embed
        self.user_embed = torch.nn.Embedding(num_embeddings=user_num_embeddings, embedding_dim=user_embed_size)
        # seed_id, history_id, and candidate_id (Item Tower) share embeddings - Youtube paper 2019[2] section 5.1 mentions sharing embeddings "among the related features"
        self.item_embed = item_embed
        # seed_artist_id, history_artist_id, and candidate_artist_id (Item Tower) share embeddings
        self.artist_embed = artist_embed
        # seed_album_id, history_album_id, and candidate_album_id (Item Tower) share embeddings
        self.album_embed = album_embed

        # calculates z_score for log(age)
        self.log_age_mean = log_age_mean
        self.log_age_std = log_age_std

        layers = []
        for i in range(len(hidden_size)):
            in_features = input_size if i == 0 else hidden_size[i-1]
            out_features = hidden_size[i]
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU())
        
        # remove last activation
        self.hidden_layers = torch.nn.Sequential(*layers[:-1])

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.xavier_normal_(module.weight)

    def forward(self, batch):
        # user id
        user_id_embedding = self.user_embed(batch["user_id"]) # (B, ) -> (B, D1)
        
        # history - last N interactions
        # Youtube paper 2019[2] section 5.1 - "We treat the watch history as a bag of words (BOW), and represent it by the average of video id embeddings."
        history_id_embedding = self.item_embed(batch["history_id"]) # (B, N) -> (B, N, D2)
        history_id_embedding = torch.mean(history_id_embedding, dim=1) # (B, N, D2) -> (B, D2)
        
        history_artist_embedding = self.artist_embed(batch["history_artist_id"]) # (B, N) -> (B, N, D3)
        history_artist_embedding = torch.mean(history_artist_embedding, dim=1) # (B, N, D3) -> (B, D3)
        
        history_album_embedding = self.album_embed(batch["history_album_id"]) # (B, N) -> (B, N, D4)
        history_album_embedding = torch.mean(history_album_embedding, dim=1) # (B, N, D4) -> (B, D4)
        
        history_track_length = torch.mean(batch["history_track_length"], dim=1).unsqueeze(1) # (B, N) -> (B, 1)
   
        # seed - Youtube paper 2019[2] Figure 2 uses seed information for context in query tower
        seed_id_embedding = self.item_embed(batch["seed_id"]) # (B, ) -> (B, D2)
        seed_artist_embedding = self.artist_embed(batch["seed_artist_id"]) # (B, ) -> (B, D3)
        seed_album_embedding = self.album_embed(batch["seed_album_id"]) # (B, ) -> (B, D4)
        seed_track_length = batch["seed_track_length"].unsqueeze(1) # (B, ) -> (B, 1)

        if self.training:
            age = (torch.log1p(batch["age"]) - self.log_age_mean) / self.log_age_std
        else:
            age = torch.zeros(
                batch["age"].shape[0],
                device=batch["age"].device,
                dtype=batch["age"].dtype,
            )
        age = age.unsqueeze(1) # (B, ) -> (B, 1)

        x = torch.cat(tensors=[user_id_embedding,         # (B, D1)
                                history_id_embedding,     # (B, D2)
                                history_artist_embedding, # (B, D3)
                                history_album_embedding,  # (B, D4)
                                history_track_length,     # (B, 1)
                                seed_id_embedding,        # (B, D2)
                                seed_artist_embedding,    # (B, D3)
                                seed_album_embedding,     # (B, D4)
                                seed_track_length,        # (B, 1)
                                age                       # (B, 1)
                            ], dim=1)
        
        out = self.hidden_layers(x)
        return out
        

class ItemTower(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: List[int], item_embed: torch.nn.Embedding, artist_embed: torch.nn.Embedding, album_embed: torch.nn.Embedding):
        super().__init__()
        
        # seed_id, history_id, and candidate_id (Item Tower) share embeddings - Youtube paper 2019[2] section 5.1 mentions sharing embeddings "among the related features"
        self.item_embed = item_embed
        # seed_artist_id, history_artist_id, and candidate_artist_id (Item Tower) share embeddings
        self.artist_embed = artist_embed
        # seed_album_id, history_album_id, and candidate_album_id (Item Tower) share embeddings
        self.album_embed = album_embed

        layers = []
        for i in range(len(hidden_size)):
            in_features = input_size if i == 0 else hidden_size[i-1]
            out_features = hidden_size[i]
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU())
        
        # remove last activation
        self.hidden_layers = torch.nn.Sequential(*layers[:-1])

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.xavier_normal_(module.weight)

    def forward(self, batch):
   
        # Candidate features - Youtube paper 2019[2] Figure 2
        candidate_id_embedding = self.item_embed(batch["candidate_id"]) # (B, ) -> (B, D2)
        candidate_artist_embedding = self.artist_embed(batch["candidate_artist_id"]) # (B, ) -> (B, D3)
        candidate_album_embedding = self.album_embed(batch["candidate_album_id"]) # (B, ) -> (B, D4)
        candidate_track_length = batch["candidate_track_length"].unsqueeze(1) # (B, ) -> (B, 1)

        x = torch.cat(tensors=[ candidate_id_embedding,        # (B, D2)
                                candidate_artist_embedding,    # (B, D3)
                                candidate_album_embedding,     # (B, D4)
                                candidate_track_length,        # (B, 1)
                            ], dim=1)
        
        out = self.hidden_layers(x)
        return out
        

class FrequencyEstimation(torch.nn.Module):
    """Youtube paper 2019[2] Algorithm 2 Implementation"""
    def __init__(self, H, alpha=0.01):
        super().__init__()
        self.H = H
        self.alpha = alpha
        # Use buffers so these are moved to GPU if the model is
        self.register_buffer('A', torch.zeros(H)) # Last step seen
        self.register_buffer('B', torch.zeros(H)) # Estimated delta (interval)
        self.register_buffer('step', torch.tensor(1.0))

    def _hash_ix(self, item_id: torch.Tensor) -> torch.Tensor:
        return item_id % self.H

    def update(self, batch_item_ids: torch.Tensor):
        # we only care about unique items in this batch for frequency
        unique_ids = torch.unique(batch_item_ids)
        h_ix = self._hash_ix(unique_ids)

        # B = (1 - alpha) * B + alpha * (current_step - last_step_seen)
        # Note: self.B[h_ix] starts at 0, so first update might be large.
        self.B[h_ix] = (1 - self.alpha) * self.B[h_ix] + self.alpha * (self.step - self.A[h_ix])
        
        # Update A with the current step
        self.A[h_ix] = self.step
        
        # Increment global step ONCE per batch
        self.step += 1

    def get_prob(self, item_ids: torch.Tensor):
        h_ix = self._hash_ix(item_ids)
        # Delta (B) is steps-between-hits. 
        # Probability p = 1 / Delta
        return 1.0 / (self.B[h_ix] + 1e-8) # Guard against division by zero
    

class TowerSolver:
    def __init__(self, 
                 query_model, 
                 item_model, 
                 data,
                 optimizer,
                 device, 
                 epochs, 
                 batch_size,
                 num_workers = 0, 
                 checkpoint_dir = "./checkpoints", 
                 checkpoint_every = 1, 
                 verbose = True, 
                 print_every = 1, 
                 reset = False, 
                 wandb_api = None,
                 H = 1000,
                 alpha = 0.01,
                 tau = 0.01
                ):
        self.query_model = query_model
        self.item_model = item_model
        self.data = data
        self.device = device
        self.epochs = epochs
        self.verbose = verbose
        self.reset = reset
        self.print_every = print_every
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.wandb_api = wandb_api
        self.tau = tau
        
        self.freq_estimator = FrequencyEstimation(H = H, alpha=alpha)
        
        self.optimizer = optimizer
        
        self.loss_history = []
        self.loss_history_batch = []

        self.freq_estimator.to(device)
        self.query_model.to(device)
        self.item_model.to(device)

        # self.wandb_run = None
        # # wandb login
        # if self.wandb_api:
        #     os.environ['WANDB_API_KEY'] = self.wandb_api
        #     wandb.login()

        #     # Start a new wandb run to track this script.
        #     self.wandb_run = wandb.init(
        #         # Set the wandb project where this run will be logged.
        #         project="Netflix-VAE",
        #         # Track hyperparameters and run metadata.
        #         config={
        #             "learning_rate": self.lr,
        #             "Optimizer": "Adam",
        #             "architecture": "VAE",
        #             "dataset": "MovieLens20M",
        #             "epochs": self.epochs,
        #             "batch_size": self.batch_size,
        #             "Max_Beta": self.anneal_cap,
        #             "Anneal_steps": self.anneal_steps,
        #             "l2_lambda": self.lam
        #         },
        #     )

        #     # Define 'epoch' as the default x-axis for all metrics
        #     wandb.define_metric("epoch", step_metric = "epoch")
    
    def _step(self):

        total_loss = 0
        nbatches = len(self.data)
        counter = 0
        for query, item, reward in self.data:
            print(f"[{counter}/{nbatches}]")
            counter += 1
            query = {k: v.to(self.device) for k, v in query.items()}
            item = {k: v.to(self.device) for k, v in item.items()}
            reward = reward.to(self.device)
            self.freq_estimator.update(item["candidate_id"])

            query_out = self.query_model(query) # (B, 128)
            item_out = self.item_model(item)    # (B, 128)

            # Youtube paper 2019[2] section 3, last paragraph (Normalization + tau)
            # l2 normalize output embeddings: embed / l2_norm(embed)
            query_out = torch.nn.functional.normalize(query_out, p=2, dim=1)
            item_out = torch.nn.functional.normalize(item_out, p=2, dim=1)

            logits = torch.matmul(query_out, item_out.T) / self.tau # (B, B)

            # Bias Correction - Youtube paper 2019[2] section 3
            # logQ correction since we are calculating batch softmax rather than full softmax
            # bias adjustment needed since more frequent items will be affected more frequenctly just because they show up more often in batch
            logits -= torch.log(self.freq_estimator.get_prob(item["candidate_id"]))
            # target is the ith candidate at the ith dot product, rest are treated as negative samples (in-batch negative sampling)
            targets = torch.arange(logits.size(0), device = logits.device)
            
            loss_per_sample = torch.nn.functional.cross_entropy(input = logits, target = targets, reduction="none") # (B, 1)
            loss = (loss_per_sample * reward / 100).mean() # scalar
            total_loss += loss.item()

            self.loss_history_batch.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return total_loss / nbatches

    
    def _save(self, loss, epoch, filepath):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint = {
            "epoch" : epoch,
            "loss" : loss,
            "query_model_state_dict": self.query_model.state_dict(),
            "item_model_state_dict": self.item_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at {filepath}")

    def _load(self, filepath):
        if not os.path.exists(filepath):
            return None, None
        checkpoint = torch.load(filepath, map_location=self.device)
        self.query_model.load_state_dict(checkpoint["query_model_state_dict"])
        self.item_model.load_state_dict(checkpoint["item_model_state_dict"])
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
        self.query_model.train()
        self.item_model.train()

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


            # if self.wandb_run:
            #     # self.wandb_run.log({"Recall@20": recall, "NDCG@20": ndcg, "Loss": loss, "epoch": epoch})
            #     self.wandb_run.log({"loss_p_batch": loss_p_batch, "loss_p_sample": loss_p_sample, "epoch": epoch})
        # if self.wandb_api:
        #     wandb.finish()