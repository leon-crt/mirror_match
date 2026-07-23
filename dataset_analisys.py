import pandas as pd
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from data_preprocessing import MatchDataset

# arguments
path = './data/Makoto2/'
flatten_folders = False

if flatten_folders:
    files = os.listdir(path)

    print("Moving all files from subfolders to main folder")
    for folder in files:
        for file in os.listdir(path + folder):
            os.rename(path + folder + '/' + file, path + file)

    print("Deleting empty sub folders")
    for folder in files:
        os.rmdir(path + folder)
    
dataset = MatchDataset(path)

x_features_mean = []
x_features_std = []

x_state_max = []
x_state_mins = []

player_inputs_count = np.zeros(12)
opponent_inputs_count = np.zeros(12)

print("Computing stats")
for x_features, y_features in dataset:
    state_feats = x_features[:,:16].numpy()
    x_features_mean.append(np.mean(state_feats,0))
    x_state_max.append(np.max(state_feats,0))
    x_state_mins.append(np.min(state_feats,0))
    for feature in y_features:
        player_inputs_count += feature.numpy()
    for feature in x_features:
        opponent_inputs_count += feature[-12:].numpy()


# Plotting input counts
input_fig, (pl, opp) = plt.subplots(1,2, sharey=True)
plt.setp((pl,opp), xticks=range(0,12), xticklabels=['left', 'up', 'right', 'down', 'lp', 'mp', 'hp', 'lk', 'mk', 'hk', 'start', 'coin'])
input_fig.set_figwidth(11)
pl.bar(range(0,12),player_inputs_count)
pl.set_title("Player")
opp.bar(range(0,12),opponent_inputs_count)
opp.set_title("Opponent")
input_fig.suptitle("Total Number of Button Presses")

total_bp_pl = player_inputs_count.sum()
probs_bp_pl = [bp/total_bp_pl for bp in player_inputs_count]

total_bp_opp = opponent_inputs_count.sum()
probs_bp_opp = [bp/total_bp_opp for bp in opponent_inputs_count]

print(f"means of the state features: {list(np.mean(x_features_mean,0))}")
print(f"max values of state features: {list(np.max(x_state_max,0))}")
print(f"min values of state features: {list(np.min(x_state_mins,0))}")
print(f"button press distribution for player inputs: {list(probs_bp_pl)}")
print(f"button press distribution for opponent inputs: {list(probs_bp_opp)}")

plt.tight_layout()
plt.show()