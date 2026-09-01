import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import numpy as np

from data_preprocessing import MatchDataset, pad_collate
from model import LSTM
from util import avg, save_checkpoint, EarlyStopping, ComputeMetrics

checkpoint_dir = 'checkpoints/'
loss_plots_dir = 'plots/'

full_dataset = MatchDataset('data/Makoto2/', autoregressive=True)
dataset_train, dataset_val = torch.utils.data.random_split(full_dataset, [0.8, 0.2])


# Neon00 Makoto2 weights torch.tensor([np.float64(3.6488536606153583), np.float64(56.805596953137936), np.float64(3.2936457761914015), np.float64(2.171915363010979), np.float64(17.536269361270556), np.float64(22.91324898445688), np.float64(51.710032010629945), np.float64(43.071203130917816), np.float64(90.91605887464125), np.float64(71.82832286733566)]

# Define hyperparameters
lstm_layers = 2
learning_rate = 1e-4
nepochs = 1000  # Maybe use loss threshold to stop automatically
batch_size = 64
positive_weights = torch.tensor([np.float64(2.9226014981326007), np.float64(11.15064121487457), np.float64(2.639903487231234), np.float64(3.260643682218815), np.float64(33.34075440214289), np.float64(35.64161551796842), np.float64(41.13497553932027), np.float64(33.98286736747931), np.float64(38.89480987215982), np.float64(86.31070670983588)])
positive_weights = torch.sqrt(positive_weights)

dl_train = DataLoader(dataset_train, batch_size, collate_fn=pad_collate)
dl_val = DataLoader(dataset_val, batch_size, collate_fn=pad_collate)

device = torch.device(0 if torch.cuda.is_available() else 'cpu')
hidden_size = 512
input_size = 36
out_size = 10 # number of pressable buttons same as targets
print(f'using device: {device}')

# Create the LSTM model
match_lstm = LSTM(input_size=input_size, output_size=out_size, hidden_size=hidden_size, num_layers=lstm_layers).to(device)
optimizer = optim.Adam(match_lstm.parameters(), lr=learning_rate)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weights).to(device)

# metrics loggers
avg_train_losses, avg_val_losses = [], []
avg_val_prec = []
avg_val_acc = []
avg_val_rec = []
avg_macro_prec, avg_macro_rec, avg_macro_acc = [], [], []

# initialize early stopping
es = EarlyStopping(min_delta=0.01, tolerance=5)

# seq have shape [batch_size, seq_len, feat_num]
# Run training loop for each epoch
for epoch in range(nepochs):
    
    print(f'Epoch: {epoch}')
    match_lstm.train()
    train_loss_logger, val_loss_logger = [], []
    val_prec_logger = []
    val_rec_logger = []
    val_acc_logger = []

    # Perform training loop
    for traj_pad, targets_pad, traj_lens, targets_lens in dl_train:
        
        # Pass the whole sequence of data at once
        traj_block = traj_pad.to(device)
        target_seq_block = targets_pad.to(device)

        hidden = torch.zeros(lstm_layers, traj_pad.shape[1], hidden_size, device=device)
        memory = torch.zeros(lstm_layers, traj_pad.shape[1], hidden_size, device=device)

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
    print('============= Training ===============')
    print(f'Average Loss Value: {avg_train_losses[-1]}')

    # perform validation loop
    match_lstm.eval()
    with torch.no_grad():
        sample_counter = 0
        for traj_pad, targets_pad, traj_lens, targets_lens in dl_val:
            
            # Pass the whole sequence of data at once
            traj_block = traj_pad.to(device)
            target_seq_block = targets_pad.to(device)
            
            # Initialize hidden state and memory, shape 1 cause 0 should be batch
            hidden = torch.zeros(lstm_layers, traj_pad.shape[1], hidden_size, device=device)
            memory = torch.zeros(lstm_layers, traj_pad.shape[1], hidden_size, device=device)

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

            # Compute Metrics
            if sample_counter % (len(dl_val) / 10) == 0:
                approx_preds = torch.sigmoid(data_pred)
                prec, rec, acc = ComputeMetrics(approx_preds, target_seq_block, out_size)
                val_prec_logger.append(prec)
                val_rec_logger.append(rec)
                val_acc_logger.append(acc)

            sample_counter += 1

        avg_val_losses.append(avg(val_loss_logger))
        avg_val_prec.append(avg(val_prec_logger))
        avg_val_rec.append(avg(val_rec_logger))
        avg_val_acc.append(avg(val_acc_logger))
        avg_macro_prec.append(avg(avg_val_prec[-1]))
        avg_macro_rec.append(avg(avg_val_rec[-1]))
        print('============= Validation ===============')
        print(f'Average Loss Value: {avg_val_losses[-1]}')
        print(f'Average Precision Value: {avg_val_prec[-1]}')
        print(f'Average Recall Value: {avg_val_rec[-1]}')
        print(f'Average Accuracy Value: {avg_val_acc[-1]}')


    if (epoch+1) % 10 == 0:
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
        plt.figure()
        plt.plot(avg_macro_prec, label='average macro precision')
        plt.plot(avg_macro_rec, label='average macro recall')
        plt.plot(avg_val_acc, label='average accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.title(f'Precision & Recall until epoch: {epoch}')
        plt.savefig(loss_plots_dir + 'metrics_' + str(epoch))
        plt.close()

    if es.early_stop(avg_val_losses[-1]):
        print('Early Stopping!')
        break
    
save_checkpoint(epoch, match_lstm, optimizer, loss, checkpoint_dir + 'checkpoint_final')
print(f'Finished training at epoch: {epoch}')