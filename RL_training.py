import warnings

import torch.optim.adam
warnings.filterwarnings("ignore")
from torch import multiprocessing
import numpy as np

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.collectors import Collector
from torchrl.modules import ProbabilisticActor, ValueOperator, LSTMModule
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import copy

from model import LSTM, Critic_LSTM, ResBlockMLP
import SF3_environment
from SF3_environment.wrappers import FlattenObservation
import gymnasium
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import GymWrapper

# TODO:
#       - Look into what to modify to have the algorithm work with multi-discrete probabilities
#       - Use log probability
#       - Split the damn LSTM into lstm and linear layers so that it can integrate into the torchRL modules nicely

ch_path = 'checkpoints/checkpoint_final'

# Hyperparameters definition
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

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

# Setting up the environment, adding flattenObservation wrapper to obtain a flat array and other wrappers for normalization and minor utils
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
env.close()

rollout = env.rollout(500)
print("rollout of three steps:", rollout)
print("Shape of the rollout TensorDict:", rollout.batch_size)

# Set up Actor and Critic networks

# HyperParameters
num_layers = 2
hidden_size = 512
input_size = 26
actor_output_size = 10
num_blocks = 1

# The models have to be dissected into their individual components so that they can interact with Tensordict nicely

# Load pre trained weights to actor and transfer them to each individual component of the model
pretrained_actor = LSTM(input_size, output_size=actor_output_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
checkpoint = torch.load(ch_path, map_location=device)
pretrained_actor.load_state_dict(checkpoint['model_state_dict'])

# Initial Input MLP
input_mlp_net = nn.Sequential(nn.Linear(input_size, 4 * input_size),
                                       nn.ReLU(),
                                       nn.Linear(4 * input_size, hidden_size))

input_mlp_net.load_state_dict(pretrained_actor.input_mlp.state_dict())

input_mlp = TensorDictModule(
    module=input_mlp_net,
    in_keys=["observation"],
    out_keys=["features"]
)

# LSTM tensordict wrapper
rec_core = LSTMModule(
    input_size=hidden_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers,
    batch_first=True,
    in_keys=["features", "rs_h", "rs_c"],
    out_keys=["rec_output", ("next", "rs_h"), ("next", "rs_c")]
)

rec_core.lstm.load_state_dict(pretrained_actor.lstm.state_dict())

# Action Head babyyyyyyy
action_head_net = nn.Sequential(*[ResBlockMLP(hidden_size, hidden_size) for _ in range(num_blocks)])

action_head = TensorDictModule(
    module=action_head_net,
    in_keys=["rec_output"],
    out_keys=["action_head_out"]
)
action_head_net.load_state_dict(pretrained_actor.res_blocks.state_dict())

# Output layer
lin_out_layer = nn.Linear(hidden_size, actor_output_size)
lin_out_layer.load_state_dict(pretrained_actor.fc_out.state_dict())
fc_out_pol_net = nn.Sequential(nn.ReLU(), lin_out_layer, nn.Sigmoid())
fc_out_pol = TensorDictModule(
    module=fc_out_pol_net,
    in_keys=["action_head_out"],
    out_keys=["value"]
)

fc_out_crit = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 1), nn.Sigmoid())

policy_module = TensorDictSequential(
    input_mlp,
    rec_core,
    action_head,
    fc_out_pol
)

value_net = TensorDictSequential(
    input_mlp,
    rec_core,
    action_head,
    fc_out_crit
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["probs"],
    distribution_class=torch.distributions.Bernoulli,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

# initialize PPO loss and advantage modules
advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=device,
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_frames // frames_per_batch, 0.0)

# initialize collector and replay buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SliceSamplerWithoutReplacement()
)

collector = Collector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()