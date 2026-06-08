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
from util import avg, save_checkpoint, EarlyStopping, ComputeMetrics

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
print(f'using device: {device}')

# Create the LSTM model
match_lstm = LSTM(input_size=28, output_size=out_size, hidden_size=hidden_size).to(device)
optimizer = optim.Adam(match_lstm.parameters(), lr=learning_rate)
loss_fn = nn.BCEWithLogitsLoss()

# metrics loggers
avg_train_losses, avg_val_losses = [], []
avg_train_prec, avg_val_prec = [], []
avg_train_rec, avg_val_rec = [], []

# initialize early stopping
es = EarlyStopping(min_delta=0.05, tolerance=5)

# seq have shape [batch_size, seq_len, feat_num]
# Run training loop for each epoch
for epoch in range(nepochs):
    
    print(f'Epoch: {epoch}')
    match_lstm.train()
    train_loss_logger, val_loss_logger = [], []
    train_prec_logger, val_prec_logger = [], []
    train_rec_logger, val_rec_logger = [], []

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
        # Compute evaluation metrics
        approx_preds = torch.sigmoid(data_pred)
        prec, rec = ComputeMetrics(approx_preds, target_seq_block, batch_size, out_size)
        
        raw_loss = loss_fn(data_pred, target_seq_block)
        mask = torch.arange(traj_pad.shape[1], device=device)[None, :] < torch.tensor(traj_lens, device=device)[:, None]
        mask = mask.unsqueeze(-1)
        masked_loss = raw_loss * mask
        loss = masked_loss.sum() / mask.sum()

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the training loss and metrics
        train_loss_logger.append(loss.item())
        train_prec_logger.append(prec)
        train_rec_logger.append(rec)

    avg_train_losses.append(avg(train_loss_logger))
    avg_train_prec.append(avg(train_prec_logger))
    avg_train_rec.append(avg(train_rec_logger))
    print('============= Training ===============')
    print(f'Average Loss Value: {avg_train_losses[-1]}')
    print(f'Average Precision Value: {avg_train_prec[-1]}')
    print(f'Average Recall Value: {avg_train_rec[-1]}')

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

            # Compute Metrics
            approx_preds = torch.sigmoid(data_pred)
            prec, rec = ComputeMetrics(approx_preds, target_seq_block, batch_size, out_size)
            
            # Log the training loss and metrics
            val_loss_logger.append(loss.item())
            val_prec_logger.append(prec)
            val_rec_logger.append(rec)

        avg_val_losses.append(avg(val_loss_logger))
        avg_val_prec.append(avg(val_prec_logger))
        avg_val_rec.append(avg(val_rec_logger))
        print('============= Validation ===============')
        print(f'Average Loss Value: {avg_val_losses[-1]}')
        print(f'Average Precision Value: {avg_val_prec[-1]}')
        print(f'Average Recall Value: {avg_val_rec[-1]}')


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
        print('Early Stopping!')
        break
    
save_checkpoint(epoch, match_lstm, optimizer, loss, checkpoint_dir + 'checkpoint_final')
print(f'Finished training at epoch: {epoch}')