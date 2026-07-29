import warnings

import torch.optim.adam
warnings.filterwarnings("ignore")
from torch import multiprocessing
import numpy as np

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, ValueOperator, LSTMModule
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import copy

from model import LSTM, Critic_LSTM
import SF3_environment
from SF3_environment.wrappers import FlattenObservation
import gymnasium
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import GymWrapper

# TODO:
#       - Look into what to modify to have the algorithm work with multi-discrete probabilities
#       - Use log probability
#       - Split the damn LSTM into lstm and linear layers so that it can integrate into the torchRL modules nicely


is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

base_env = gymnasium.make("SF3_environment/StreetFighter3-v0", render_mode="human", mode="cpu")
flat_env = FlattenObservation(base_env)

torch_env = GymWrapper(flat_env)

env = TransformedEnv(
    torch_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)

env.transform[0].init_stats(num_iter=1000)
env.close()

print("normalization constant shape:", env.transform[0].loc.shape)
print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("input_spec:", env.input_spec)
print("action_spec (as defined by input_spec):", env.action_spec)

check_env_specs(env)

frames_per_batch = 3000
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

# Set up Actor and Critic networks
in_size = 28
action_num = 12
hidden_size = 128

actor_instance = LSTM(in_size, output_size=action_num, hidden_size=hidden_size).to(device)
critic_instance = Critic_LSTM(input_size=in_size, output_size=1, hidden_size=hidden_size).to(device)

# load model checkpoint
ch_path = 'checkpoints/checkpoint_273_Harmonaz'
checkpoint = torch.load(ch_path, map_location=device)
actor_instance.load_state_dict(checkpoint['model_state_dict'])

# initialize torchrl wrappers to properly integrate with tensordict and other ppo stuff
actor_lstm = LSTMModule(
    lstm=actor_instance.lstm
    in_keys=["state", "act_h", "act_c"],
    out_keys=["actions", ("next", "act_h"), ("next", "act_c")]
)
critic_lstm = LSTMModule(
    lstm=critic_instance.lstm,
    in_keys=["state", "cri_h", "cri_c"],
    out_keys=["value", ("next", "cri_h"), ("next", "cri_c")]
)
policy_module = TensorDictModule(actor_instance, in_keys=["observation"], out_keys=["logits"])
policy_module = ProbabilisticActor(
    policy_module,
    in_keys=["logits"],
    spec=env.action_spec,
    distribution_class=torch.distributions.

    )

# initialize PPO modules
adv_module = GAE(gamma=gamma, lmbda=lmbda, value_network=critic_lstm, average_gae=True, device=device)
loss_module = ClipPPOLoss(
    actor_network=actor_lstm,
    critic_network=critic_lstm,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coeff=entropy_eps
)
optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_frames // frames_per_batch, 0.0)
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SliceSamplerWithoutReplacement()
)

for _ in range(total_frames // frames_per_batch):
    # Collect trajectories
    batch = get_trajectories_vs_CPU(actor_instance, hidden_size, device, frame_num=frames_per_batch) # use actor or actor_instance here?
    for _ in range(num_epochs):
        # Calculate Advantage
        adv_module(batch)
        # update replay buffer with potential newly collected data, advantage calculations and critic values calculated in the inner loop
        replay_buffer.extend(batch)
        for _ in range(frames_per_batch // sub_batch_size):
            sub_batch = replay_buffer.sample(sub_batch_size)
            # loss module takes care of calculating the values using the critic and applying everything else to calculate loss
            loss_vals = loss_module(sub_batch.to(device))
            loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )
            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()
    scheduler.step()