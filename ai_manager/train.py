import numpy as np
import torch
from config import config


class TimeSeriesTrainer:
  def __init__(self, model, loaders, criterion, optimizer, **kwargs):
    self.model = model
    self.loaders = loaders
    self.criterion = criterion
    self.optimizer = optimizer
    self.model_type = config.get('model', 'MODEL_TYPE').lower()
    self.num_epochs = config.getint('model', 'NUM_EPOCHS')
    self.best_loss = float('inf')
    self.early_stop = kwargs.get('early_stop', True)
    self.is_train = config.getboolean('model', 'IS_TRAIN')

    self.train_map = {
      'predictor': self._train_predictor,
      'reconstructor': self._train_reconstructor
    }


  def train(self):
    if not self.is_train:
      print(f"Set 'IS_TRAIN' as 'True' if you want to train your model.")
      return None
    called_fn = self.train_map[self.model_type]
    if called_fn:
      called_fn()
    else:
      raise ValueError(f"Wrong Model Type: {self.model_type}. Choose Between 'reconstructor' and 'predictor'.")
    self._save_model()
    return self.model


  def _train_predictor(self):
    for epoch in range(self.num_epochs):
      self.model.train()
      train_loss = 0.0

      for X_batch, y_batch in self.loaders['train']:
        y_pred = self.model(X_batch)
        loss = self.criterion(y_pred, y_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_loss += loss.item()

      avg_train_loss = train_loss / len(self.loaders['train'])
      avg_eval_loss, rmse = self._evaluate(is_reconstructor=False)
      print(f'[Epoch {epoch+1}] train_loss={avg_train_loss:.4f}, eval_loss={avg_eval_loss:.4f}, RMSE={rmse:.4f}')
      if self._check_early_stop(avg_eval_loss):
        break


  def _train_reconstructor(self):
    for epoch in range(self.num_epochs):
      self.model.train()
      train_loss = 0.0

      for X_batch in self.loaders['train']:
        X_pred = self.model(X_batch)
        loss = self.criterion(X_pred, X_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_loss += loss.item()

      avg_train_loss = train_loss / len(self.loaders['train'])
      avg_eval_loss, rmse = self._evaluate(is_reconstructor=True)
      print(f'[Epoch {epoch+1}] train_loss={avg_train_loss:.4f}, eval_loss={avg_eval_loss:.4f}, RMSE={rmse:.4f}')
      if self._check_early_stop(avg_eval_loss):
        break


  def _evaluate(self, is_reconstructor=True):
    self.model.eval()
    eval_loss = 0.0

    with torch.no_grad():
      for batch in self.loaders['train']:
        if is_reconstructor:
          X = batch
          y = batch
        else:
          X = batch[0]
          y = batch[1]
        
        pred = self.model(X)
        eval_loss += self.criterion(pred, y).item()
    
    avg_loss = eval_loss / len(self.loaders['train'])
    return avg_loss, np.sqrt(avg_loss)


  def _check_early_stop(self, current_loss):
    if current_loss < self.best_loss:
      self.best_loss = current_loss
      return False
    else:
      if self.early_stop:
        print(f'Early Stop - best_loss: {self.best_loss:.4f}, current_loss: {current_loss:.4f}')
        return True
      return False
    
  def _save_model(self):
    model_name = config.get('model', 'MODEL_NAME')
    torch.save(self.model.state_dict(), f'{model_name}.pth')