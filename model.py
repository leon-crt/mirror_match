import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ResBlockMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlockMLP, self).__init__()
        # Define layers for the MLP block
        self.norm1 = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, input_size//2)
        self.norm2 = nn.LayerNorm(input_size//2)
        self.fc2 = nn.Linear(input_size//2, output_size)
        self.fc3 = nn.Linear(input_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        # Forward pass through the MLP block
        x = self.act(self.norm1(x))
        skip = self.fc3(x)  # Skip connection
        x = self.act(self.norm2(self.fc1(x)))
        x = self.fc2(x)
        return x + skip

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=1, hidden_size=128, num_layers=1):
        super(LSTM, self).__init__()
        # Define layers for input MLP, LSTM, residual blocks, and output linear layer
        self.input_mlp = nn.Sequential(nn.Linear(input_size, 4 * input_size),
                                       nn.ReLU(),
                                       nn.Linear(4 * input_size, hidden_size))
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        blocks = [ResBlockMLP(hidden_size, hidden_size) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, input_seq:torch.Tensor, hidden_in, mem_in, pad_seqs_lens = None, act_last_layer = False):
        # Pass input sequence through the input MLP and pack it
        if not act_last_layer: # If the pad_seqs_lens is None then we are in inference mode
            input_vec = pack_padded_sequence(self.input_mlp(input_seq), pad_seqs_lens, batch_first=True, enforce_sorted=False)
        else:
            input_vec = self.input_mlp(input_seq)
        # Pass the input MLP output through the LSTM block
        output, (hidden_out, mem_out) = self.lstm(input_vec, (hidden_in, mem_in))
        if not act_last_layer:
            output, _ = pad_packed_sequence(output, batch_first=True)
            # Pass the LSTM output through residual blocks
            x = self.act(self.res_blocks(output)) 
            # Pass the output of the residual blocks through the final linear layer
            return self.fc_out(x), hidden_out, mem_out # we are training so do not apply sigmoid as it will already be in loss
        else:
            x = self.act(self.res_blocks(output))
            # Pass the output of the residual blocks through the final linear layer
            return torch.sigmoid(self.fc_out(x)), hidden_out, mem_out # we are in inference mode so apply sigmoid to output
        
class Critic_LSTM(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=1, hidden_size=128):
        super(Critic_LSTM, self).__init__()
        # Define layers for input MLP, LSTM, residual blocks, and output linear layer
        self.input_mlp = nn.Sequential(nn.Linear(input_size, 4 * input_size),
                                       nn.ReLU(),
                                       nn.Linear(4 * input_size, hidden_size))
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        blocks = [ResBlockMLP(hidden_size, hidden_size) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, input_seq:torch.Tensor, hidden_in, mem_in):
        # Pass input sequence through the input MLP
        input_vec = self.input_mlp(input_seq)
        # Pass the input MLP output through the LSTM block
        output, (hidden_out, mem_out) = self.lstm(input_vec, (hidden_in, mem_in))
        x = self.act(self.res_blocks(output))
        # Pass the output of the residual blocks through the final linear layer
        return self.fc_out(x), hidden_out, mem_out

def reward(prev_state, new_state):
    r = 0
    health_pl = 2
    health_opp = 10
    meter_pl = 3
    stun_pl = 5
    stun_opp = 13
    got_hit = False
    
    if prev_state == None:
        return 0

    # check health values
    if new_state[health_pl] == 0: # terminal failure (player lost)
        r = -10
    elif new_state[health_opp] == 0: # terminal success (opponent lost)
        r = 10
    if prev_state[health_pl] > new_state[health_pl]: # player got hit
        r -= 0.3
        got_hit = True
    if prev_state[health_opp] > new_state[health_opp]: # opponent got hit
        r += 0.3
    # time-step penalty
    if new_state[health_pl] <= new_state[health_opp]:
        r -= 0.05
    
    # stun related rewards and penalties
    if new_state[stun_pl] == 1 and prev_state[stun_pl] == 0:
        r -= 0.5
    if new_state[stun_opp] == 1 and prev_state[stun_opp] == 0:
        r += 0.5
    
    # meter related rewards and penalties
    if new_state[meter_pl] > prev_state[meter_pl] and (not got_hit):
        r += 0.1
    
    return r