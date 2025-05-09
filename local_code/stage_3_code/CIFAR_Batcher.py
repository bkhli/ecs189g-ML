

import torch
from torch.utils.data import Dataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class CIFAR_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32).to(device=device) / 255.0  # normalize
        self.y = torch.tensor(np.array(y), dtype=torch.long).to(device=device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
