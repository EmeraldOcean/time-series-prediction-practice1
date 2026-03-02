import torch
import torch.nn as nn
from config import config


class LSTMModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.input_size=config.getint('model', 'INPUT_SIZE')
    self.hidden_size=config.getint('model', 'HIDDEN_SIZE')
    self.output_size=config.getint('model', 'OUTPUT_SIZE')

    self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
    self.linear = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, x):
    x, _ = self.lstm(x)
    x = x[:, -1, :]  # output_size로 조정
    x = self.linear(x)
    return x
  
  def load_model(self):
    model_path = config.get('model', 'MODEL_NAME')
    model = LSTMModel()
    model.load_state_dict(torch.load(f'{model_path}.pth', weights_only=True))
    return model