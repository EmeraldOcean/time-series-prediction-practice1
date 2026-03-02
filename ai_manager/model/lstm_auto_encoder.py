import torch
import torch.nn as nn
from config import config


class Encoder(nn.Module):
  def __init__(self, input_size=4096, hidden_size=1024, num_layers=2, device='cpu'):
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.device = device

    self.lstm = nn.LSTM(
        input_size, hidden_size, num_layers,
        batch_first=True, dropout=0.1, bidirectional=False
        ).to(device)

  def forward(self, x):
    x = x.to(self.device)
    outputs, (hidden, cell) = self.lstm(x)
    return hidden, cell


class Decoder(nn.Module):
  def __init__(self, input_size=4096, hidden_size=1024, output_size=4096, num_layers=2, device='cpu'):
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_layers = num_layers
    self.device = device

    self.lstm = nn.LSTM(
       input_size, hidden_size, num_layers,
        batch_first=True, dropout=0.1, bidirectional=False
        ).to(device)

    self.fc = nn.Linear(hidden_size, output_size).to(device)

  def forward(self, x, hidden):
    x = x.to(self.device)
    output, (hidden, cell) = self.lstm(x, hidden)
    prediction = self.fc(output)
    return prediction, (hidden, cell)


class LSTMAutoEncoder(nn.Module):
  def __init__(self):
    super(LSTMAutoEncoder, self).__init__()
    self.input_dim = config.getint('model', 'INPUT_SIZE')
    self.latent_dim = config.getint('model', 'LATENT_SIZE')
    
    self.window_size = config.getint('model', 'WINDOW_SIZE')
    self.device = config.get('model', 'MODEL_DEVICE')
    self.num_layers = config.getint('model', 'NUM_LAYERS')

    self.encoder = Encoder(
      input_size=self.input_dim,
      hidden_size=self.latent_dim,
      num_layers=self.num_layers,
      device=self.device
      )

    self.reconstruct_decoder = Decoder(
      input_size=self.input_dim,
      output_size=self.input_dim,
      hidden_size=self.latent_dim,
      num_layers=self.num_layers,
      device=self.device
      )

    self.to(self.device)

  def forward(self, src):
    src = src.to(self.device)
    batch_size, sequence_length, var_length = src.size()

    hidden = self.encoder(src)

    inv_idx = torch.arange(sequence_length - 1, -1, -1).long().to(self.device)

    reconstruct_output = []
    temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(self.device)

    for t in range(sequence_length):
      temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
      reconstruct_output.append(temp_input)

    reconstruct_output = torch.cat(reconstruct_output, dim=1)[:, inv_idx, :]
    return reconstruct_output
  
  def load_model(self):
    model_path = config.get('model', 'MODEL_NAME')
    model = LSTMAutoEncoder()
    model.load_state_dict(torch.load(f'{model_path}.pth', weights_only=True))
    return model