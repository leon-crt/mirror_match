import numpy as np
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch

MAX_X = 928
MIN_X = 93
MAX_Y = 226
MIN_Y = -42

class MatchDataset(Dataset):
    def __init__(self, dir_path, autoregressive=False):
        self.dir_path = dir_path
        self.file_names = os.listdir(dir_path)
        self.autoregressive = autoregressive

        # extract max values for normalization
        max_posX = np.zeros((len(self.file_names)))
        max_posY = np.zeros((len(self.file_names)))
        max_stun = np.zeros((len(self.file_names)))
        max_meter = np.zeros((len(self.file_names)))
        min_posX = np.zeros((len(self.file_names)))
        min_posY = np.zeros((len(self.file_names)))
        len_seqs = np.zeros((len(self.file_names)))
        for i in range(len(self.file_names)):
            raw_data = pd.read_csv(self.dir_path + self.file_names[i])
            len_seqs[i] = len(raw_data)
            max_posX[i] = raw_data['PosX'].max()
            max_posY[i] = raw_data['PosY'].max()
            max_stun[i] = raw_data['Stun'].max()
            max_meter[i] = raw_data['Meter'].max()
            min_posY[i] = raw_data['PosY'].min()
            min_posX[i] = raw_data['PosX'].min()
        self.max_posX = max_posX.max()
        self.min_posX = min_posX.min()
        self.max_posY = max_posY.max()
        self.min_posY = min_posY.min()
        self.max_stun = max_stun.max()
        self.max_meter = max_meter.max()
        self.max_seq_len = int(len_seqs.max())

    def __len__(self):
        return len(self.file_names)

    def norm_scalar_features(self, player_data):
        player_data['PosX'] = (player_data['PosX'] - MIN_X) / (MAX_X- MIN_X)
        player_data['PosY'] = (player_data['PosY'] - MIN_Y) / (MAX_Y - MIN_Y)
        player_data['Health'] = player_data['Health'] / player_data['Health'].max()
        player_data['Meter'] = player_data['Meter'] / self.max_meter
        player_data['Stun'] = player_data['Stun'] / self.max_stun
        return player_data
    
    def __getitem__(self, idx):
        filename = self.file_names[idx]
        raw_data = pd.read_csv(self.dir_path + filename)
        player_side = filename[0]
        p1_data = raw_data.loc[raw_data.index % 2 == 0].reset_index()
        p2_data = raw_data.loc[raw_data.index % 2 == 1].reset_index()
        target_data, opponent_data = p1_data, p2_data
        if int(player_side) == 2:
            target_data = p2_data
            opponent_data = p1_data
            
        labels = target_data[['Left','Up','Right','Down','Lp','Mp','Hp','Lk','Mk','Hk']].copy()

        # normalize scalar features
        target_data = self.norm_scalar_features(target_data)
        opponent_data = self.norm_scalar_features(opponent_data)

        if not self.autoregressive:
            states = pd.concat([target_data[['PosX','PosY','Health','Meter','Stun','isStunned','Hit','Thrown']], opponent_data[['PosX','PosY','Health','Meter','Stun','isStunned','Hit','Thrown','Left','Up','Right','Down','Lp','Mp','Hp','Lk','Mk','Hk']]], axis=1)
        else: 
            target_data_next = target_data[1:-1]
            states = pd.concat([target_data[['PosX','PosY','Health','Meter','Stun','isStunned','Hit','Thrown']], target_data[['Left','Up','Right','Down','Lp','Mp','Hp','Lk','Mk','Hk']], opponent_data[['PosX','PosY','Health','Meter','Stun','isStunned','Hit','Thrown','Left','Up','Right','Down','Lp','Mp','Hp','Lk','Mk','Hk']]], axis=1)
                
        return torch.tensor(states.values, dtype=torch.float32), torch.tensor(labels.values, dtype=torch.float32)

    def get_max_seq_len(self):
        return self.max_seq_len

# This is necessary because all the trajectories from replays have differing length but we still want to process them in batches 
def pad_collate(batch):
    (trajectories, targets) = zip(*batch)
    traj_lens = [len(seq) for seq in trajectories]
    targets_lens = [len(seq) for seq in targets]

    traj_pad = pad_sequence(trajectories, batch_first=True, padding_value=0)
    targets_pad = pad_sequence(targets, batch_first=True, padding_value=0)

    return traj_pad, targets_pad, traj_lens, targets_lens