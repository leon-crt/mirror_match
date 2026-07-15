import warnings

import torch.optim.adam
warnings.filterwarnings("ignore")
from torch import multiprocessing
import numpy as np

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import copy

from model import LSTM, Critic_LSTM
from SF3_environment import get_trajectories_vs_CPU

def get_gaes(rewards, dones, values, gamma = 0.99, lamda = 0.9, normalize=True):
    deltas = np.zeros(len(rewards))
    for i in reversed(range(len(rewards))):
        deltas[i] = rewards[i] + gamma * (1 - dones[i]) * values[i+1] - values[i]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

    target = gaes + values[:-1].detach().numpy()
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    return np.vstack(gaes), np.vstack(target)

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 50_000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

# Set up environment here
traj_num = 4
scheduler_max_it = 1000

# Set up Actor and Critic networks
in_size = 28
action_num = 12
hidden_size = 128

actor = LSTM(in_size, output_size=action_num, hidden_size=hidden_size).to(device)
critic = Critic_LSTM(input_size=in_size, output_size=1, hidden_size=hidden_size).to(device)

# load model checkpoint
ch_path = 'checkpoints/checkpoint_273_Harmonaz'
checkpoint = torch.load(ch_path, map_location=device)
actor.load_state_dict(checkpoint['model_state_dict'])

# Collect trajectories
trajectories = get_trajectories_vs_CPU(actor, hidden_size, device, round_num=traj_num)

dones = np.zeros(len(trajectories[0][2]))
dones[-1] = 1

# Collect values for the trajectories from the critic network
t_values = []
hidden = torch.zeros(1, 1, hidden_size, device=device)
memory = torch.zeros(1, 1, hidden_size, device=device)
for traj in trajectories:
    c_input = traj[0].unsqueeze(0)
    out, hidden, memory = critic(c_input, hidden, memory)
    t_values.append(out)

# Calculate Advantage
adv, critic_target = get_gaes(trajectories[0][2], dones, t_values[0].squeeze(0).squeeze(1))
adv_module = GAE(gamma=gamma, lmbda=lmbda, value_network=critic, average_gae=True, device=device)
loss_module = ClipPPOLoss(
    actor_network=actor,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coeff=entropy_eps
)
optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, scheduler_max_it, 0.0)

# Optimize Critic Network


# Optimize Actor Network