import torch
from torch.utils.data import Dataset


class AEWindowDataset(Dataset):
  def __init__(self, data, windows):
    self.data = data
    self.windows = windows

  def __len__(self):
    return len(self.windows)

  def __getitem__(self, idx):
    start, end = self.windows[idx]
    values = self.data[start:end]
    x = torch.tensor(values, dtype=torch.float32)
    return x


class LSTMWindowDataset(Dataset):
  def __init__(self, data, windows):
    self.data = data
    self.windows = windows

  def __len__(self):
    return len(self.windows)

  def __getitem__(self, idx):
    start, end = self.windows[idx]
    values = self.data[start:end]
    window = torch.tensor(values, dtype=torch.float32)
    x = window[:-1]
    y = window[-1]
    return x, y