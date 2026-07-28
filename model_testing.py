import SF3_environment as SF3Env
from SF3_environment.wrappers import FlattenObservation
import torch
from model import LSTM
from util import format_pred_env, normalize
import gymnasium

# lstm init
in_size = 28
hidden_size = 128
out_size = 12
num_layers = 1
model = LSTM(input_size=in_size, output_size=out_size, num_layers=num_layers, hidden_size=hidden_size)
hidden = torch.zeros([num_layers, 1, hidden_size])
memory = torch.zeros([num_layers, 1, hidden_size])
threshold = 0.49

device = torch.device("cpu")
ch_path = 'checkpoints/checkpoint_final'
checkpoint = torch.load(ch_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# environment init
base_env = gymnasium.make("SF3_environment/StreetFighter3-v0", render_mode="human", mode="free")
env = FlattenObservation(base_env)

state, _ = env.reset()

while(True):
    state = normalize(state)
    state = torch.tensor(state, dtype=torch.float32)
    out, hidden, memory = model(state.unsqueeze(0).unsqueeze(0), hidden, memory, act_last_layer=True)
    action = format_pred_env(out.reshape(-1), threshold)
    state, reward, done, _, _ = env.step(action) 

