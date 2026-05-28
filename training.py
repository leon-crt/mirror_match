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
from util import avg, save_checkpoint, EarlyStopping

# TODO: 
#       - implement early stopping based on val loss [test]
#       - implement the ground truth being the autoregressive lstm input

checkpoint_dir = 'checkpoints/'
loss_plots_dir = 'plots/'

full_dataset = MatchDataset('data/Makoto2/Akuma1/train/')
dataset_train, dataset_val = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

dl_train = DataLoader(dataset_train, 32, collate_fn=pad_collate)
dl_val = DataLoader(dataset_val, 32, collate_fn=pad_collate)

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
avg_train_losses = []
avg_val_losses = []

# initialize early stopping
es = EarlyStopping(min_delta=10, tolerance=5)

# seq have shape [batch_size, seq_len, feat_num]
# Run training loop for each epoch
for epoch in range(nepochs):
    
    match_lstm.train()
    train_loss_logger = []
    val_loss_logger = []

    # Perform training loop
    for traj_pad, targets_pad, traj_lens, targets_lens in dl_train:
        
        # Pass the whole sequence of data at once
        traj_block = traj_pad.to(device)
        target_seq_block = targets_pad.to(device)

        hidden = torch.zeros(1, traj_pad.shape[1], hidden_size, device=device)
        memory = torch.zeros(1, traj_pad.shape[1], hidden_size, device=device)

        # Pass the input sequence through the LSTM
        data_pred, _, _ = match_lstm(traj_block, hidden, memory, traj_lens)
        
        # Calculate the loss
        raw_loss = loss_fn(data_pred, target_seq_block)
        mask = torch.arange(traj_pad.shape[1], device=device)[None, :] < torch.tensor(traj_lens, device=device)[:, None]
        mask = mask.unsqueeze(-1)
        masked_loss = raw_loss * mask
        loss = masked_loss.sum() / mask.sum()

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log the training loss
        train_loss_logger.append(loss.item())

    avg_train_losses.append(avg(train_loss_logger))

    # perform validation loop
    match_lstm.eval()
    with torch.no_grad():
        for traj_pad, targets_pad, traj_lens, targets_lens in dl_val:
            
            # Pass the whole sequence of data at once
            traj_block = traj_pad.to(device)
            target_seq_block = targets_pad.to(device)
            
            # Initialize hidden state and memory, shape 1 cause 0 should be batch
            hidden = torch.zeros(1, traj_pad.shape[1], hidden_size, device=device)
            memory = torch.zeros(1, traj_pad.shape[1], hidden_size, device=device)

            # Pass the input sequence through the LSTM
            data_pred, _, _ = match_lstm(traj_block, hidden, memory, traj_lens)
            
            # Calculate the loss
            raw_loss = loss_fn(data_pred, target_seq_block)
            mask = torch.arange(traj_pad.shape[1], device=device)[None, :] < torch.tensor(traj_lens, device=device)[:, None]
            mask = mask.unsqueeze(-1)
            masked_loss = raw_loss * mask
            loss = masked_loss.sum() / mask.sum()

            # Log the training loss
            val_loss_logger.append(loss.item())

        avg_val_losses.append(avg(val_loss_logger))


    if (epoch+1) % 100 == 0:
        # save weight checkpoint
        filename = 'checkpoint_' + str(epoch)
        save_checkpoint(epoch, match_lstm, optimizer, loss, checkpoint_dir + filename)
        # make and save loss function plot
        plt.figure()
        plt.plot(avg_train_losses, label='average train. loss')
        plt.plot(avg_val_losses, label='average val. loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title('Loss at epoch ' + str(epoch))
        plt.savefig(loss_plots_dir + 'loss_' + str(epoch))
        plt.close()
    
    if es.early_stop(avg_val_losses[-1]):
        break
    
save_checkpoint(nepochs, match_lstm, optimizer, loss, checkpoint_dir + 'checkpoint_final')
