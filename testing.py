import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_preprocessing import MatchDataset, pad_collate
from model import LSTM

checkpoint_path = 'checkpoints/checkpoint_final'
dataset_test = MatchDataset('data/Makoto2/Akuma1/test')
dl_test = DataLoader(dataset_test, 32, collate_fn=pad_collate)

device = torch.device(0 if torch.cuda.is_available() else 'cpu')
hidden_size = 128
out_size = 12
match_lstm = LSTM(input_size=28, output_size=out_size, hidden_size=hidden_size).to(device)

loss_fn = nn.MSELoss()
testing_loss_logger = []

# load the model
checkpoint = torch.load(checkpoint_path)
match_lstm.load_state_dict(checkpoint['model_state_dict'])

# evaluate
match_lstm.eval()

for traj_pad, targets_pad, traj_lens, targets_lens in dl_test:
    # Pass the whole sequence of data at once
    traj_block = traj_pad.to(device)
    target_seq_block = targets_pad.to(device)
    
    # Initialize hidden state and memory, shape 1 cause 0 should be batch
    hidden = torch.zeros(1, traj_pad.shape[1], hidden_size, device=device)
    memory = torch.zeros(1, traj_pad.shape[1], hidden_size, device=device)

    # Pass the input sequence through the LSTM
    data_pred, hidden, memory = match_lstm(traj_block, hidden, memory, traj_lens)
    
    # Calculate the loss
    loss = loss_fn(data_pred, target_seq_block)

    # Log the training loss
    testing_loss_logger.append(loss.item())

