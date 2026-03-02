from data import DataManager, set_dataloader
from ai_manager import ModelManager, TimeSeriesTrainer, TimeSeriesEvaluation, AnomalyMetrics
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
  data_manager = DataManager()
  all_dataset = data_manager.prepare_dataset()
  data_loaders = set_dataloader(all_dataset)
  model_manager = ModelManager()
  model = model_manager.get_model()

  criterion = nn.MSELoss()
  lr = 5e-4
  optimizer = optim.Adam(model.parameters(), lr=lr)

  trainer = TimeSeriesTrainer(model, data_loaders, criterion, optimizer)
  trainer.train()

  loaded_model = model.load_model()
  prediction = TimeSeriesEvaluation(loaded_model, data_loaders)
  threshold, results = prediction.evaluate_model()
  anomaly_metrics = AnomalyMetrics(results, threshold)
  anomaly_metrics.plot_metrics('./result')