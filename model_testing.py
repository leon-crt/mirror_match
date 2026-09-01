import SF3_environment as SF3Env
from SF3_environment.wrappers import FlattenObservation
import torch
import matplotlib.pyplot as plt
from model import LSTM
import numpy as np
from util import format_pred_env, normalize
from rl_util import transpose_weights_nn_to_rl
import gymnasium

# lstm init
in_size = 36
hidden_size = 512
out_size = 10
num_layers = 2
model = LSTM(input_size=in_size, output_size=out_size, num_layers=num_layers, hidden_size=hidden_size)
hidden = torch.zeros([num_layers, 1, hidden_size])
memory = torch.zeros([num_layers, 1, hidden_size])
threshold = 0.49
RL_weights = False
autoreg = True

device = torch.device("cpu")
ch_path = 'checkpoints/checkpoint_209'
checkpoint = torch.load(ch_path, map_location=device) 
if RL_weights:
    model = transpose_weights_nn_to_rl(checkpoint, model)
else:
    model.load_state_dict(checkpoint['model_state_dict'])

# environment init
base_env = gymnasium.make("SF3_environment/StreetFighter3-v0", render_mode="human", mode="free")
env = FlattenObservation(base_env)

state, _ = env.reset()
log_rewards = []
done = False
out = torch.zeros((10,))

while(True):
    state = normalize(state)
    state = torch.tensor(state, dtype=torch.float32)
    if autoreg:
        state = torch.concat((state, out))
    out, hidden, memory = model(state.unsqueeze(0).unsqueeze(0), hidden, memory, act_last_layer=True)
    out = out.reshape(-1)
    action = format_pred_env(out, threshold)
    state, reward, done, _, _ = env.step(action) 
    log_rewards.append(reward)

env.close()
plt.plot(log_rewards)
plt.show()
log_rewards = np.array(log_rewards)
print(f"cumulative rewards: {log_rewards.sum()}")
print(f"average reward: {log_rewards.mean()}")
