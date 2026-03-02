import numpy as np
import pandas as pd
from config import config
from .window_dataset import AEWindowDataset, LSTMWindowDataset


DATASET_MAP = {
  'predictor': LSTMWindowDataset,
  'reconstructor': AEWindowDataset
}

class DataManager:
  def __init__(self):
    data_path = config.get('data', 'DATA_PATH')
    self.df = pd.read_csv(data_path)
    self.mean = None
    self.std = None
    self.window_size = config.getint('model', 'WINDOW_SIZE')

    self.model_type = config.get('model', 'MODEL_TYPE').lower()


  def prepare_dataset(self):
    self.df = self._select_df_columns()
    split_results = self._split_train_test_data()

    train_df, threshold_df, test_df, abnormal_df = split_results

    scaled_data_list = [
      self._scale_data(train_df, is_train=True),
      self._scale_data(threshold_df),
      self._scale_data(test_df),
      self._scale_data(abnormal_df)
    ]

    windows_list = [
      self._calculate_windows(train_df),
      self._calculate_windows(threshold_df),
      self._calculate_windows(test_df),
      self._calculate_windows(abnormal_df)
    ]

    dataset_list = []
    for idx, value in zip(windows_list, scaled_data_list):
      dataset_list.append(self._create_dataset(value, idx))

    all_dataset = {
      'train': dataset_list[0],
      'threshold': dataset_list[1],
      'test': {
        'normal': dataset_list[2],
        'abnormal': dataset_list[3]
      }
    }

    return all_dataset


  def _select_df_columns(self):
    columns_list = config.get('data', 'COLUMNS_LIST')
    if (type(columns_list) is str):
      columns_list = columns_list.split(',')
    elif (type(columns_list) is list):
      pass
    return self.df[columns_list]


  def _split_train_test_data(self):
    n_train = config.getint('data', 'N_TRAIN')
    n_threshold = config.getint('data', 'N_THRESHOLD')
    label_column = config.get('data', 'LABEL_COLUMN')

    normal_df = self.df[self.df[label_column] == 0].copy()  # 정상 라벨은 0, 비정상 라벨은 1로 고정
    normal_df.drop([label_column], axis=1, inplace=True)
    abnormal_df = self.df[self.df[label_column] == 1].copy()
    abnormal_df.drop([label_column], axis=1, inplace=True)

    train_df = normal_df.iloc[: n_train]
    threshold_df = normal_df.iloc[n_train : n_train+n_threshold]
    test_df = normal_df.iloc[n_train+n_threshold :]

    return train_df, threshold_df, test_df, abnormal_df


  def _scale_data(self, data, is_train=False):
    apply_norm = config.getboolean('data', 'APPLY_NORM')
    array_data = data.values if isinstance(data, pd.DataFrame) else np.array(data)

    if is_train:
      self.mean = array_data.mean(axis=0)
      self.std = array_data.std(axis=0)
      self.std[self.std == 0] = 1e-8  # divide-by-zero 방지

    if apply_norm:
      return (array_data - self.mean) / self.std
    else:
      return array_data
    
  
  def _calculate_windows(self, df):
    windows = []
    for i in range(0, len(df)-self.window_size+1):
      start_idx = df.index[i]
      end_idx = df.index[i+self.window_size-1]

      if (end_idx - start_idx) == (self.window_size - 1):
        windows.append((i, i+self.window_size))
    return windows


  def _create_dataset(self, data, windows):
    dataset_class = DATASET_MAP.get(self.model_type)
    if dataset_class is None:
      raise ValueError(f"Wrong Model Type: {self.model_type}. Choose Between 'reconstructor' and 'predictor'.")
    return dataset_class(data, windows)