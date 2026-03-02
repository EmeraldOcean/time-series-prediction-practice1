import numpy as np
import torch
from config import config


class TimeSeriesEvaluation:
  def __init__(self, model, loaders):
    self.model = model
    self.threshold_loader = loaders['threshold']
    self.test_loaders = loaders['test'] if isinstance(loaders['test'], dict) else {'test':loaders['test']}
    self.threshold = None
    self.test_results = {}
    self.model_type = config.get('model', 'MODEL_TYPE').lower()
    self.model.eval()

    self.calculate_map = {
      'predictor': self._calculate_predictor,
      'reconstructor': self._calculate_reconstructor
    }

  def evaluate_model(self):
    self._calculate_threshold()
    self._calculate_test()

    return self.threshold, self.test_results


  def _calculate_threshold(self, percentile=95):
    called_fn = self.calculate_map[self.model_type]
    if called_fn:
      threshold_mse_scores = called_fn(self.threshold_loader)
    else:
      raise ValueError(f"Wrong Model Type: {self.model_type}. Choose Between 'reconstructor' and 'predictor'.")

    self.threshold = np.percentile(threshold_mse_scores, percentile)
    return self.threshold


  def _calculate_test(self):
    for name, loader in self.test_loaders.items():
      called_fn = self.calculate_map[self.model_type]
      if called_fn:
        scores = called_fn(loader)
      else:
        raise ValueError(f"Wrong Model Type: {self.model_type}. Choose Between 'reconstructor' and 'predictor'.")
      self.test_results[name] = scores
    return self.test_results


  def _calculate_predictor(self, loader):
    mse_scores = []
    with torch.no_grad():
      for x, y in loader:
        pred = self.model(x)
        mse = torch.mean((y - pred)**2, dim=1)
        mse_scores.append(mse)
    return torch.cat(mse_scores, dim=0).numpy()


  def _calculate_reconstructor(self, loader):
    mse_scores = []
    with torch.no_grad():
      for x in loader:
        pred = self.model(x)
        mse = torch.mean((x - pred)**2, dim=(1,2))
        mse_scores.append(mse)
    return torch.cat(mse_scores, dim=0).numpy()