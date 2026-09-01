import torch
import subprocess
import socket
import win32process
import win32gui
import win32api
import win32con
from tensordict import TensorDict
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch import nn
from model import ResBlockMLP

from model import ResBlockMLP

MAX_X = 928
MIN_X = 93
MAX_Y = 226
MIN_Y = -42

HOST = "127.0.0.1"
PORT = 42069

# lstm initialization
device = torch.device(0 if torch.cuda.is_available() else 'cpu')
hidden_size = 512
out_size = 10 # number of pressable buttons same as targets
threshold = 0.3
input_size = 26
num_layers = 2
num_blocks= 1

class SelfPlayLSTMWrapper(gym.Wrapper):
    def __init__(self, env, lstm_model, hidden_size, num_layers, threshold):
        super().__init__(env)
        self.model = lstm_model
        self.hidden = torch.zeros(num_layers, 1, hidden_size)
        self.memory = torch.zeros(num_layers, 1, hidden_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.thresh = threshold
        self.current_observation = torch.zeros(26, dtype=torch.float32)
        self.previous_player_action = np.zeros(10, dtype=np.float32)
        self.previous_opp_action = torch.zeros((10,))
        
        # Set obs space
        mins = np.array([-np.inf] * 16 + [0] * 10, dtype=np.float32)
        maxs = np.array([np.inf] * 16 + [1] * 10, dtype=np.float32)

        self.observation_space = Box(low=mins, high=maxs, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        self.memory = torch.zeros(self.num_layers, 1, self.hidden_size)
        
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Format and force float32 array output
        obs = np.concatenate([obs["player_state"], obs['opponent_state'], obs['opponent_inputs']]).astype(np.float32)
        self.current_observation = torch.from_numpy(normalize(obs)).float()
        self.previous_player_action = np.zeros(10, dtype=np.float32)
        self.previous_opp_action = torch.zeros((10,))
        
        return obs, info

    def step(self, action_player):
        # Convert previous action to float32 tensor matching self.current_observation
        prev_act_tensor = torch.from_numpy(self.previous_player_action).float()

        opp_obs = torch.cat((self.current_observation[8:16], self.current_observation[:8], prev_act_tensor, self.previous_opp_action))
        
        # Predict opponent action
        out, self.hidden, self.memory = self.model(
            opp_obs.unsqueeze(0).unsqueeze(0), 
            self.hidden, 
            self.memory, 
            act_last_layer=True
        )
        out = out.reshape(-1)
        self.previous_opp_action = out
        action_opponent = format_pred_env(out, self.thresh)
        
        # Step base environment
        kwargs = {
            "action_player": action_player,
            "action_opp": action_opponent
        }
        obs, reward, done, trunc, info = self.env.unwrapped.step(**kwargs)
        
        obs = np.concatenate([obs["player_state"], obs['opponent_state'], obs['opponent_inputs']]).astype(np.float32)
        self.current_observation = torch.from_numpy(normalize(obs)).float()
        self.previous_player_action = np.array(action_player, dtype=np.float32)
        
        return obs, reward, done, trunc, info

def normalize(obs):
    MAX_X = 928
    MIN_X = 93
    MAX_Y = 226
    MIN_Y = -42

    obs[0] = (obs[0] - MIN_X) / (MAX_X - MIN_X)
    obs[8] = (obs[8] - MIN_X) / (MAX_X - MIN_X)
    obs[1] = (obs[1] - MIN_Y) / (MAX_Y - MIN_Y)
    obs[9] = (obs[9] - MIN_Y) / (MAX_Y - MIN_Y)
    obs[2] = obs[2] / 161
    obs[10] = obs[10] / 161
    obs[3] = obs[3] / 161
    obs[11] = obs[11] / 336
    obs[4] = obs[4] / 161
    obs[12] = obs[12] / 70
    return obs

def enumWindowsProc(hwnd, lParam):
    if (lParam is None) or ((lParam is not None) and win32process.GetWindowThreadProcessId(hwnd)[1] == lParam):
        text = win32gui.GetWindowText(hwnd)
        if text:
            win32api.SendMessage(hwnd, win32con.WM_CLOSE)

def save_checkpoint(epoch, model, optimizer, loss, path):
    checkpoint = {
        'epoch': epoch + 1, # Save the next epoch number to start from
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss, # Or maybe validation loss
        }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} to {path}")

def avg(l):
    sum = 0
    for i in l:
        sum += i
    return sum / len(l)

def format_pred(t:torch.Tensor, threshold):
    res = [None] * len(t)
    for i in range(len(t)):
        if t[i] > threshold:
            res[i] = 1
        else:
            res[i] = 0

    return str(res)[1:-1].replace(' ', '')

def format_pred_env(t:torch.Tensor, threshold):
    res = [None] * len(t)
    for i in range(len(t)):
        if t[i] > threshold:
            res[i] = 1
        else:
            res[i] = 0

    return res

class EarlyStopping():
    def __init__(self, min_delta, tolerance):
        self.min_delta = min_delta
        self.tolerance = tolerance
        self.min_validation_loss = float('inf')
        self.counter = 0

    def early_stop(self, val_loss):
        if val_loss < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_validation_loss + self.min_delta):
            self.counter+=1
            if self.counter >= self.tolerance:
                return True
        return False
    
def ComputeMetrics(pred_seq: torch.Tensor, target_seq, out_size):
    pred_np = pred_seq.cpu().detach().numpy()
    target_np = target_seq.cpu().detach().numpy()
    
    # 2. Flatten 3D sequential data [batch, seq, classes] -> 2D matrix [samples, classes]
    # This resolves the "unknown is not supported" error
    pred_flat = pred_np.reshape(-1, out_size)
    target_flat = target_np.reshape(-1, out_size).astype(int)
    
    # 3. Correct thresholding order (Assuming pred_seq contains probabilities)
    # Convert decimals to 0 or 1, THEN cast to integer
    pred_binary = (pred_flat >= 0.5).astype(int)
    prec = precision_score(target_flat, pred_binary, average=None, zero_division=0)
    rec = recall_score(target_flat, pred_binary, average=None, zero_division=0)
    acc = accuracy_score(target_flat, pred_binary)
    
    return prec, rec, acc

class PlayerFeatures():
    def __init__(self, posx, posy, health, meter, stun, isStunned, hit, thrown, inputs):
        self.posX = posx
        self.posY = posy
        self.health = health
        self.meter = meter
        self.stun = stun
        self.hit = hit
        self.isStunned = isStunned
        self.thrown = thrown
        self.inputs = inputs
    
    def normalize(self):
        self.posX = (self.posX - MIN_X) / (MAX_X - MIN_X)
        self.posY = (self.posY - MIN_Y) / (MAX_Y - MIN_Y)
        self.health = self.health / 161
        self.meter = self.meter / 336
        self.stun = self.stun / 70

class GameState():
    def __init__(self, s:str):
        self.feats = s.split(sep=',')[-29:-1]
        self.feats = [int(x) for x in self.feats]
        self.P1 = PlayerFeatures(*self.feats[:8], None)
        self.P2 = PlayerFeatures(*self.feats[8:16], self.feats[16:])
    
    def normalize(self):
        self.P1.normalize()
        self.P2.normalize()
        # p1 features
        self.feats[0] = self.P1.posX
        self.feats[1] = self.P1.posY
        self.feats[2] = self.P1.health
        self.feats[3] = self.P1.meter
        self.feats[4] = self.P1.stun
        # p2 features
        self.feats[8] = self.P2.posX
        self.feats[9] = self.P2.posY
        self.feats[10] = self.P2.health
        self.feats[11] = self.P2.meter
        self.feats[12] = self.P2.stun