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
from torchrl.modules import ProbabilisticActor, ValueOperator, LSTMModule
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
from util import EarlyStopping, SelfPlayLSTMWrapper
from rl_util import transpose_weights_nn_to_rl, MaskInitState, InitZeroState

class IndependentBernoulli(Independent):
    def __init__(self, probs=None, logits=None):
        base_dist = Bernoulli(probs=probs, logits=logits)
        super().__init__(base_dist, reinterpreted_batch_ndims=1)

class CriticHead(nn.Module):
    def __init__(self, hid_size, mem_size, num_blocks, num_layers_lstm):
        super(CriticHead, self).__init__()
        input_size = (hid_size + mem_size) * num_layers_lstm
        blocks = [ResBlockMLP(input_size, input_size) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)
        self.fc1 = nn.Linear(input_size, input_size//2)
        self.out = nn.Linear(input_size//2, 1)
        self.act = nn.LeakyReLU()

    def forward(self, hid_state, mem_state):
        input_state = torch.cat((hid_state, mem_state), 1)
        x = self.act(self.res_blocks(input_state))
        x = self.act(self.fc1(x))
        return self.out(x)

ch_path = 'checkpoints/Harmonaz-tf-512-2/checkpoint_249'
plots_dir = "plots/RL/"
rl_weights = False

# Hyperparameters definition
frames_per_batch = 5000
# For a complete training, bring the number of frames up to 1M
total_frames = 1_000_000

sub_batch_size = 500  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.85
entropy_eps = 1e-10
temperature = 1.0

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-6
max_grad_norm = 1.0

# HyperParameters
num_layers = 2
hidden_size = 512
input_size = 36
actor_output_size = 10
num_blocks = 1

# Load pre trained weights to actor and transfer them to each individual component of the model
pretrained_actor = LSTM(input_size, output_size=actor_output_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
checkpoint = torch.load(ch_path, map_location=device)
if rl_weights:
    pretrained_actor = transpose_weights_nn_to_rl(checkpoint, pretrained_actor)
else:
    pretrained_actor.load_state_dict(checkpoint['model_state_dict'])

# Setting up the environment, adding flattenObservation wrapper to obtain a flat array and other wrappers for normalization and minor utils
base_env = gymnasium.make("SF3_environment/StreetFighter3-v0", render_mode="turbo", mode="selfplay")
# Create and load opponent model
self_play_env = SelfPlayLSTMWrapper(base_env, pretrained_actor, hidden_size, num_layers, 0.49)

torch_env = GymWrapper(self_play_env)

env = TransformedEnv(
    torch_env,
    Compose(
        ObservationNorm(in_keys=["observation"]),
        InitZeroState(keys=["actor_prev_output", "critic_prev_output"], feature_dims=[actor_output_size, 1]),
        InitTracker(),
        StepCounter(),
    ),
)

env.transform[0].init_stats(num_iter=frames_per_batch*3)

print("normalization constant shape:", env.transform[0].loc.shape)
print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("input_spec:", env.input_spec)
print("action_spec (as defined by input_spec):", env.action_spec)

check_env_specs(env)
# Set up Actor and Critic networks
# The models have to be dissected into their individual components so that they can interact with Tensordict nicely

def recurrent_body(prefix, input_size=36, state_dict_mlp=None, state_dict_lstm=None):
    reset_prev_out = TensorDictModule(
        module=MaskInitState(),
        in_keys=[f"{prefix}_prev_output", "is_init"],
        out_keys=[f"{prefix}_prev_output_clean"],
    )

    cat_module = CatTensors(
        in_keys=["observation", f"{prefix}_prev_output_clean"],
        out_key=f"{prefix}_cat_input",
        dim=-1,
        del_keys=False
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

action_head = TensorDictModule(
    module=action_head_net,
    in_keys=["actor_features"],
    out_keys=["action_head_out"]
)
action_head_net.load_state_dict(pretrained_actor.res_blocks.state_dict())

# Output layer
lin_out_layer = nn.Linear(hidden_size, actor_output_size)
lin_out_layer.load_state_dict(pretrained_actor.fc_out.state_dict())
fc_out_pol_net = nn.Sequential(nn.ReLU(), lin_out_layer)

class TemperatureModule(nn.Module):
    def __init__(self, temp):
        super(TemperatureModule, self).__init__()
        self.temp = temp

    def forward(self, logits):
        return logits / self.temp

temp_module = TensorDictModule(
    module=TemperatureModule(temperature),
    in_keys=["raw_logits"],
    out_keys=["logits"]
)

fc_out_pol = TensorDictModule(
    module=fc_out_pol_net,
    in_keys=["action_head_out"],
    out_keys=["raw_logits"]
)

class SigmoidModule(nn.Module):
    def forward(self, logits):
        return torch.sigmoid(logits)

actor_feedback_module = TensorDictModule(
    module=SigmoidModule(),
    in_keys=["logits"],
    out_keys=[("next", "actor_prev_output")],
)

actor_rec = recurrent_body("actor", state_dict_mlp=pretrained_actor.input_mlp.state_dict(), state_dict_lstm=pretrained_actor.lstm.state_dict())

policy_module = ProbabilisticActor(
    module=TensorDictSequential(
        actor_rec,
        action_head,
        fc_out_pol,
        temp_module,
        actor_feedback_module
    ),
    spec=env.action_spec,
    in_keys=["logits"],
    distribution_class=IndependentBernoulli,
    return_log_prob=True,
)

critic_feedback_module = TensorDictModule(
    module=nn.Identity(),
    in_keys=["critic_features"],
    out_keys=[("next", "critic_prev_output")]
)

value_module = TensorDictSequential(
    recurrent_body("critic", input_size=27),
    critic_feedback_module,
    ValueOperator(nn.Linear(hidden_size, 1), in_keys=["critic_features"]),
)

if rl_weights:
    value_module.load_state_dict(checkpoint['critic_state_dict'])

print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

# initialize PPO loss and advantage modules
advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=device, deactivate_vmap=True,
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coeff=entropy_eps,
    critic_coeff=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_frames // frames_per_batch, 0.0)

# initialize collector and replay buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch),
    sampler=SliceSamplerWithoutReplacement(num_slices=1, strict_length=False),
    batch_size=sub_batch_size
)

collector = Collector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
    auto_register_policy_transforms=True,
)

logs = defaultdict(list)
actor_frozen = True
policy_module.requires_grad_(False)
pbar = tqdm(total=total_frames)
eval_str = ""
critic_loss_target = 0.007

# Freeze policy module until critic is up to speed
# policy_module.requires_grad_(False)
# grad_pol = False
# es = EarlyStopping(min_delta=0.1, tolerance=3)

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for epoch in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        epoch_loss = defaultdict(list)
        replay_buffer.extend(tensordict_data.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )
            epoch_loss["loss_objective"].append(loss_vals["loss_objective"].detach().numpy())
            epoch_loss["loss_critic"].append(loss_vals["loss_critic"].detach().numpy())
            epoch_loss["loss_entropy"].append(loss_vals["loss_entropy"].detach().numpy())

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

            # Unfreeze this mf actor
            # if not grad_pol and es.early_stop(loss_vals["loss_critic"]):
            #     policy_module.requires_grad_(True)

    mean_critic_loss = np.mean(epoch_loss["loss_critic"])
    logs["loss_critic"].append(mean_critic_loss)

    # Check if critic loss has reached the target threshold
    if actor_frozen and mean_critic_loss < critic_loss_target:
        policy_module.requires_grad_(True)
        actor_frozen = False
        print(f"\n[Iteration {i}] Critic loss reached {mean_critic_loss:.4f}. Unfreezing actor.")

    logs["loss_objective"].append(np.array(epoch_loss["loss_objective"]).mean())
    logs["loss_critic"].append(np.array(epoch_loss["loss_critic"]).mean())
    logs["loss_entropy"].append(np.array(epoch_loss["loss_entropy"]).mean())
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
        # Save parameters
        filename = 'checkpoint_' + str(i)
        checkpoint = {
            "epoch": epoch + 1,
            'model_state_dict': policy_module.state_dict(),
            'critic_state_dict': value_module.state_dict(),
            'loss': loss_value
        }

        torch.save(checkpoint, f'./checkpoints/RL/{filename}')
        plt.plot(logs["loss_objective"], label="objective loss")
        plt.plot(logs["loss_critic"], label="critic loss")
        plt.plot(logs["loss_entropy"], label="entropy loss")
        plt.xlabel("Steps")
        plt.ylabel("Losses")
        plt.legend()
        plt.savefig(plots_dir + 'loss_' + str(i))
        plt.close()
        plt.plot(logs['reward'])
        plt.xlabel("Epochs")
        plt.ylabel("Reward")
        plt.savefig(plots_dir + 'reward_' + str(i))
        plt.close()
        


    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()

# Save parameters
filename = 'checkpoint_' + str(i) + '_final'
checkpoint = {
    "epoch": epoch + 1,
    'model_state_dict': policy_module.state_dict(),
    'critic_state_dict': value_module.state_dict(),
    'loss': loss_value
}

torch.save(checkpoint, f'./checkpoints/RL/{filename}')