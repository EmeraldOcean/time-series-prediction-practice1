from .model.lstm import LSTMModel
from .model.lstm_auto_encoder import LSTMAutoEncoder
from config import config

class ModelManager:
  def __init__(self):
    self.model_name = config.get('model', 'MODEL_NAME').lower()
    self.model = None

    self.model_map = {
      'lstm': LSTMModel,
      'lstm-auto-encoder': LSTMAutoEncoder
    }

    model_class = self.model_map[self.model_name]
    if model_class:
      self.model = model_class()
    else:
      raise ValueError(f"""
          Wrong Model Name: {self.model_name}. Choose among below list.
          - lstm
          - lstm-auto-encoder
          """)

  def get_model(self):
    return self.model