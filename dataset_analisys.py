import pandas as pd
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from data_preprocessing import MatchDataset

# arguments
path = './data/Akuma1/'
flatten_folders = True

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

player_inputs_count = np.zeros(10)
opponent_inputs_count = np.zeros(10)

left_side = 0
right_side = 0

total_frames = 0
file_counter = 0
errors = []

print("Computing stats")
for x_features, y_features in dataset:
    pl_posx = x_features[:,0].numpy()
    opp_posx = x_features[:,8].numpy()
    for i in range(len(pl_posx)):
        if pl_posx[i] > opp_posx[i]:
            right_side += 1
        else:
            left_side += 1

    state_feats = x_features[:,:16].numpy()
    x_features_mean.append(np.mean(state_feats,0))
    x_state_max.append(np.max(state_feats,0))
    x_state_mins.append(np.min(state_feats,0))
    total_frames += len(y_features)
    for feature in y_features:
        player_inputs_count += feature.numpy()
    for feature in x_features:
        opponent_inputs_count += feature[-10:].numpy()
    file_counter += 1

# Plotting input counts
input_fig, (pl, opp) = plt.subplots(1,2, sharey=True)
plt.setp((pl,opp), xticks=range(0,10), xticklabels=['left', 'up', 'right', 'down', 'lp', 'mp', 'hp', 'lk', 'mk', 'hk'])
input_fig.set_figwidth(11)
pl.bar(range(0,10),player_inputs_count)
pl.set_title("Player")
opp.bar(range(0,10),opponent_inputs_count)
opp.set_title("Opponent")
input_fig.suptitle("Total Number of Button Presses")

total_bp_pl = player_inputs_count.sum()
probs_bp_pl = [bp/total_bp_pl for bp in player_inputs_count]

total_bp_opp = opponent_inputs_count.sum()
probs_bp_opp = [bp/total_bp_opp for bp in opponent_inputs_count]

num_neg_samples_per_class = np.array([total_frames] * 10) - player_inputs_count
pos_weights = num_neg_samples_per_class / player_inputs_count

# Plotting playing side
side_fig, pls = plt.subplots()
pls.bar(range(0,2), [left_side,right_side])
pls.set_xticks([0,1], ['Left', 'Right'])
pls.set_title("Player side presence throughout all frames")


print(f"means of the state features: {list(np.mean(x_features_mean,0))}")
print(f"max values of state features: {list(np.max(x_state_max,0))}")
print(f"min values of state features: {list(np.min(x_state_mins,0))}")
print(f"button press distribution for player inputs: {list(probs_bp_pl)}")
print(f"button press distribution for opponent inputs: {list(probs_bp_opp)}")
print(f"weighting for loss function: {list(pos_weights)}")

plt.tight_layout()
plt.show()