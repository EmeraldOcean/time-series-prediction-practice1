from torch.utils.data import DataLoader
from config import config

def set_dataloader(datasets):
  loaders = {}
  batch_size=config.getint('model', 'BATCH_SIZE')

  for key, value in datasets.items():
    if (isinstance(value, dict)):
      loaders[key] = set_dataloader(value)
    else:
      loaders[key] = DataLoader(value, batch_size=batch_size, shuffle=False)  # 시계열 데이터이므로 shuffle=False
  return loaders