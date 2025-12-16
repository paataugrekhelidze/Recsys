import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from scipy.sparse import csr_matrix
import random

class UserItemDataset(Dataset):
    """takes User-Item interaction Dataframe, returns a sample of userId, itemId and rating triplet"""
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.users = data["userId"].values
        self.movies = data["movieId"].values
        self.ratings = data["rating"].values
    def __len__(self):
        return len(self.users)
    def __getitem__(self, ix):
        return self.users[ix], self.movies[ix], float(self.ratings[ix])

class UserItemDatasetSparse(Dataset):
    """takes User-Item interaction Dataframe, returns a sample of userId, itemId and rating triplet"""
    def __init__(self, data: pd.DataFrame, p: float = 0.1):
        super().__init__()
        self.users = data["userId"].values
        self.movies = data["movieId"].values
        self.ratings = data["rating"].values
        self.max_items = self.movies.max()
        self.p = p
        self.sparse_matrix = csr_matrix((self.ratings, (self.users, self.movies)))
    def __len__(self):
        return len(self.users)
    def __getitem__(self, ix):
        if random.random() < self.p:
            return self.users[ix], self.movies[ix], int(self.ratings[ix])
        userId = int(self.users[ix])
        movieId = random.randint(0, self.max_items)
        return userId, movieId, int(self.sparse_matrix[userId, movieId])
    
class WeightedMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super().__init__()

        # (U x D)
        self.user_embeddings = nn.Embedding(num_embeddings=num_users, embedding_dim=num_factors)
        # (V x D)
        self.item_embeddings = nn.Embedding(num_embeddings=num_items, embedding_dim=num_factors)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        
    def forward(self, user_ix, item_ix):
        """
        takes user and item indices, returns dot product value, one for each pair of user-item embedding

        Inputs:
         - user_ix: user indices, of shape
              (N, 1), where N is the batch size
         - item_ix: item indices, of shape
              (N, 1), where N is the batch size
        Returns:
         - out: dot product of user and item embedding pair that correspond to the respective indices, of shape
              (N, 1), where N is the batch size   
        """
        user_ix_embeddings = self.user_embeddings(user_ix)
        item_ix_embeddings = self.item_embeddings(item_ix)
        return (user_ix_embeddings * item_ix_embeddings).sum(dim=1)
        
        