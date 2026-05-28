import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from data_preprocessing import MatchDataset, pad_collate
from model import LSTM

# TODO: 
#       - display the loss
#       - create a test dataset and investigate if it's possible to have cross-validation
#       - implement self stopping based on loss
#       - implement the ground truth being the autoregressive lstm input

dataset_train = MatchDataset('data/Makoto2/Akuma1/')
dl_train = DataLoader(dataset_train, 32, collate_fn=pad_collate)
# Define hyperparameters
learning_rate = 1e-4 
nepochs = 500  # Maybe use loss threshold to stop automatically
batch_size = 32

device = torch.device(0 if torch.cuda.is_available() else 'cpu')
hidden_size = 128
out_size = 12 # number of pressable buttons same as targets

# Create the LSTM model
match_lstm = LSTM(input_size=28, output_size=out_size, hidden_size=hidden_size).to(device)
optimizer = optim.Adam(match_lstm.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
training_loss_logger = []

# Remember to use pack_padded_sequence before feeding input to LSTM in training loop https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
# seq have shape [batch_size, seq_len, feat_num]

# Run training loop for each epoch
for epoch in range(nepochs):
    # Set the model to training mode
    match_lstm.train()
    
    # Iterate through the training data loader
    for traj_pad, targets_pad, traj_lens, targets_lens in dl_train:
        
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
        
        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log the training loss
        training_loss_logger.append(loss.item())