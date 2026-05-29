import numpy as np
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch

class MatchDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.file_names = os.listdir(dir_path)

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
        self.min_posX = min_posY.min()
        self.max_posY = max_posY.max()
        self.min_posY = min_posX.min()
        self.max_stun = max_stun.max()
        self.max_meter = max_meter.max()
        self.max_seq_len = int(len_seqs.max())

    def __len__(self):
        return len(self.file_names)

    def norm_scalar_features(self, player_data):
        player_data['PosX'] = player_data['PosX'] - self.min_posX / self.max_posX - self.min_posX
        player_data['PosY'] = player_data['PosY'] - self.min_posY / self.max_posY - self.min_posY
        player_data['Health'] = player_data['Health'] / player_data['Health'].max()
        player_data['Meter'] = player_data['Meter'] / self.max_meter
        player_data['Stun'] = player_data['Stun'] / self.max_stun
        return player_data
    
    def __getitem__(self, idx):
        raw_data = pd.read_csv(self.dir_path + self.file_names[idx])
        p1_data = raw_data.loc[raw_data['Player'] == 'P1'].reset_index()
        p2_data = raw_data.loc[raw_data['Player'] == 'P2'].reset_index()
        targets = p1_data[['Left','Up','Right','Down','Lp','Mp','Hp','Lk','Mk','Hk','Start','Coin']].copy()

        # normalize scalar features
        p1_data = self.norm_scalar_features(p1_data)
        p2_data = self.norm_scalar_features(p2_data)
        
        states = pd.concat([p1_data[['PosX','PosY','Health','Meter','Stun','isStunned','Hit','Thrown']], p2_data[['PosX','PosY','Health','Meter','Stun','isStunned','Hit','Thrown','Left','Up','Right','Down','Lp','Mp','Hp','Lk','Mk','Hk','Start','Coin']]], axis=1)
    
        return torch.tensor(states.values, dtype=torch.float32), torch.tensor(targets.values, dtype=torch.float32)

    def get_max_seq_len(self):
        return self.max_seq_len

# This is necessary because all the trajectories from replays have differing length but we still want to process them in batches 
# TODO: batch is fucked for some reason it's only 9 long and it has weird values
def pad_collate(batch):
    (trajectories, targets) = zip(*batch)
    traj_lens = [len(seq) for seq in trajectories]
    targets_lens = [len(seq) for seq in targets]

    traj_pad = pad_sequence(trajectories, batch_first=True, padding_value=0)
    targets_pad = pad_sequence(targets, batch_first=True, padding_value=0)

    return traj_pad, targets_pad, traj_lens, targets_lens