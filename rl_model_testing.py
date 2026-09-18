import warnings

import torch.optim.adam
warnings.filterwarnings("ignore")
from torch import multiprocessing
import numpy as np

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv, InitTracker)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.collectors import Collector
from tensordict.nn import TensorDictModule, TensorDictSequential 
from torchrl.modules import ProbabilisticActor, ValueOperator, LSTMModule, set_recurrent_mode
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import copy
from torch.distributions import Bernoulli, Independent
import gymnasium
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import GymWrapper
from torchrl.envs.transforms import CatTensors

from model import LSTM, ResBlockMLP
import SF3_environment
from SF3_environment.wrappers import FlattenObservation
from util import EarlyStopping, SelfPlayLSTMWrapper, format_pred
from rl_util import transpose_weights_nn_to_rl, MaskInitState, InitZeroState, NormalizeObs

class IndependentBernoulli(Independent):
    def __init__(self, probs=None, logits=None):
        base_dist = Bernoulli(probs=probs, logits=logits)
        super().__init__(base_dist, reinterpreted_batch_ndims=1)

ch_path = 'checkpoints/checkpoint_final'
rl_weights = False

# For a complete training, bring the number of frames up to 1M
total_frames = 10000

clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.85
entropy_eps = 0.0

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

# HyperParameters
num_layers = 2
hidden_size = 512
input_size = 36
actor_output_size = 10
num_blocks = 1

# Load pre trained weights to actor and transfer them to each individual component of the model
checkpoint = torch.load(ch_path, map_location=device)

pretrained_actor = None
if not rl_weights:
    pretrained_actor = LSTM(input_size, output_size=actor_output_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    pretrained_actor.load_state_dict(checkpoint['model_state_dict'])

# Setting up the environment, adding flattenObservation wrapper to obtain a flat array and other wrappers for normalization and minor utils
base_env = gymnasium.make("SF3_environment/StreetFighter3-v0", render_mode="human", mode="free")
base_env = FlattenObservation(base_env)
base_env = NormalizeObs(base_env)
# Create and load opponent model

torch_env = GymWrapper(base_env)

env = TransformedEnv(
    torch_env,
    Compose(
        InitZeroState(keys=["actor_prev_output"], feature_dims=[actor_output_size]),
        InitTracker(),
        StepCounter(),
    ),
)

print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("input_spec:", env.input_spec)
print("action_spec (as defined by input_spec):", env.action_spec)

# Set up Actor and Critic networks
# The models have to be dissected into their individual components so that they can interact with Tensordict nicely

class InputCat(nn.Module):
    def forward(self, observation, actor_prev_output_clean):
        catInp = torch.cat((observation, actor_prev_output_clean),-1)
        return catInp

def recurrent_body(prefix, input_size=36, state_dict_mlp=None, state_dict_lstm=None):
    reset_prev_out = TensorDictModule(
        module=MaskInitState(),
        in_keys=[f"actor_prev_output", "is_init"],
        out_keys=[f"actor_prev_output_clean"],
    )

    cat_module = TensorDictModule(
        module=InputCat(),
        in_keys=["observation", "actor_prev_output_clean"],
        out_keys=f"{prefix}_cat_input",
    )

    input_mlp = TensorDictModule(
                module=nn.Sequential(
                    nn.Linear(input_size, 4*input_size),
                    nn.ReLU(),
                    nn.Linear(4 * input_size, hidden_size)
                ),
                in_keys=[f"{prefix}_cat_input"],
                out_keys=[f"{prefix}_embed"]
            )
    LSTM = LSTMModule(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                in_keys=[f"{prefix}_embed", f"{prefix}_rs", f"{prefix}_rc", "is_init"],
                out_keys=[f"{prefix}_features", ("next", f"{prefix}_rs"), ("next", f"{prefix}_rc")],
                recurrent_backend="auto",
            )

    if state_dict_lstm != None:
        LSTM.lstm.load_state_dict(state_dict_lstm)
    if state_dict_mlp != None:
        input_mlp.module.load_state_dict(state_dict_mlp)

    return TensorDictSequential(
        reset_prev_out,
        cat_module,
        input_mlp,
        LSTM,
    )

action_head_net = nn.Sequential(*[ResBlockMLP(hidden_size, hidden_size) for _ in range(num_blocks)])
lin_out_layer = nn.Linear(hidden_size, actor_output_size)
actor_rec = None
if not rl_weights:
    action_head_net.load_state_dict(pretrained_actor.res_blocks.state_dict())
    lin_out_layer.load_state_dict(pretrained_actor.fc_out.state_dict())
    actor_rec = recurrent_body("actor", state_dict_mlp=pretrained_actor.input_mlp.state_dict(), state_dict_lstm=pretrained_actor.lstm.state_dict())
else:
    actor_rec = recurrent_body("actor")

action_head = TensorDictModule(
    module=action_head_net,
    in_keys=["actor_features"],
    out_keys=["action_head_out"]
)

# Output layer
fc_out_pol_net = nn.Sequential(nn.ReLU(), lin_out_layer, nn.Sigmoid())

fc_out_pol = TensorDictModule(
    module=fc_out_pol_net,
    in_keys=["action_head_out"],
    out_keys=["probs"]
)

class DebugModule(nn.Module):
    def forward(self, probs, observation, actor_prev_output_clean, actor_cat_input):
        probs = probs
        return probs

actor_feedback_module = TensorDictModule(
    module=DebugModule(),
    in_keys=["probs", "observation", "actor_prev_output_clean", "actor_cat_input"],
    out_keys=[("next", "actor_prev_output")],
)

policy_module = ProbabilisticActor(
    module=TensorDictSequential(
        actor_rec,
        action_head,
        fc_out_pol,
        actor_feedback_module
    ),
    spec=env.action_spec,
    in_keys=["probs"],
    distribution_class=IndependentBernoulli,
    return_log_prob=True,
)

if rl_weights:
    policy_module.load_state_dict(checkpoint["model_state_dict"])

collector = Collector(
    env,
    policy_module,
    frames_per_batch=total_frames,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
    auto_register_policy_transforms=True,
)

for i, tensordict_data in enumerate(collector):
    break