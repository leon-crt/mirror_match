import torch
import subprocess
import socket
import win32process
import win32gui
import win32api
import win32con
from tensordict import TensorDict
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

from model import LSTM, reward

MAX_X = 928
MIN_X = 93
MAX_Y = 226
MIN_Y = -42

HOST = "127.0.0.1"
PORT = 42069

# lstm initialization
device = torch.device(0 if torch.cuda.is_available() else 'cpu')
hidden_size = 128
out_size = 12 # number of pressable buttons same as targets
threshold = 0.3

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

def get_trajectories_vs_CPU(model, hidden_size, device, frame_num, CPU=True):
    
    hidden = torch.zeros(1, 1, hidden_size, device=device)
    memory = torch.zeros(1, 1, hidden_size, device=device)
    batch = TensorDict({
        "states": torch.zeros(frame_num, 28),
        "actions": torch.zeros(frame_num, 12),
        "rewards": torch.zeros(frame_num, 1),
        "dones": torch.zeros(frame_num, 1)
    },
    batch_size=[frame_num])
    current_frame = 0

    
    while current_frame < frame_num:
        # start emulator
        emu_proc = subprocess.Popen(["./fbneo/fcadefbneo.exe", "sfiii3nr1", "./model_game_interface.lua"])

        round_counter = 0
        match_finished = False
        match_start_frame = current_frame
        # establish tcp connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST,PORT))
            s.settimeout(10)
            # game loop
            while not match_finished and (current_frame < frame_num):
                try:
                    s.listen()
                    conn, addr = s.accept()
                    with conn:
                        while current_frame < frame_num:
                            try:
                                data = conn.recv(100)
                                # catch graceful disconnection
                                if not data:
                                    print(f"Client {addr} disconnected gracefully.")
                                    break
                                
                                data = data.decode('utf-8')

                                # if data contains 'R' then the round is over and we store the trajectory before moving on to the next
                                if data[-1] == 'R':
                                    round_counter += 1
                                    data = data.replace('#', '')
                                    print(data)
                                    state = GameState(data[:-1])
                                    print(state.feats)
                                    state.normalize()
                                    # compute reward
                                    previous_state = batch["states"][current_frame - 1]
                                    r = reward(previous_state, state.feats)
                                    # update main variables
                                    batch["rewards"][current_frame] = r
                                    batch["dones"][current_frame] = 1
                                    batch["states"][current_frame] = torch.tensor(state.feats)
                                    batch["actions"][current_frame] = batch["actions"][current_frame-1] # duplicate last model output so states and actions have the same length
                                    # Send the player inputs for the last frame otherwise the lua script throws a tantrum
                                    conn.send(bytes(format_pred(torch.zeros(12), threshold) + '\r\n', "utf-8"))
                                    
                                    # if 2 rounds have been played then count it as a finished match and start another instance of emulator
                                    if round_counter >= 2:
                                        match_finished = True
                                        round_counter = 0
                                        break
                                else:
                                    # decode the string and remove the padding
                                    data = data.replace('#', '')
                                    state = GameState(data)
                                    state.normalize()

                                    # calculate the reward and update environment variables
                                    if current_frame - match_start_frame > 0:
                                        previous_state = batch["states"][current_frame - 1]
                                        r = reward(previous_state, state.feats)
                                        batch["rewards"][current_frame] = r
                                    batch["states"][current_frame] = torch.tensor(state.feats)
                                    
                                    # feed to lstm
                                    lstm_input = torch.tensor(state.feats, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                                    player_inputs, hidden, memory = model(lstm_input, hidden, memory, act_last_layer=True)
                                    player_inputs = player_inputs.squeeze(0).squeeze(0)
                                    batch["actions"][current_frame] = player_inputs
                                    # send through the socket as a comma separated list of numbers
                                    conn.send(bytes(format_pred(player_inputs, threshold) + '\r\n', "utf-8"))
                                
                                current_frame += 1
                            except (ConnectionResetError, BrokenPipeError) as e:
                                # catch abrupt network cut
                                print(f"Connection with {addr} was interrupted abruptly: {e}")
                                break
                except(TimeoutError):
                    print('TIMEOUTTTTT')
                    break
            
        win32gui.EnumWindows(enumWindowsProc, emu_proc.pid)

    return batch

def free_play(model):
    
    hidden = torch.zeros(1, 1, hidden_size, device=device)
    memory = torch.zeros(1, 1, hidden_size, device=device)

    # start emulator
    subprocess.Popen(["./fbneo/fcadefbneo.exe", "sfiii3nr1", "./sf3_env.lua"])

    # establish tcp connection
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST,PORT))
        s.settimeout(10)
        # game loop
        while True:
            try:
                s.listen()
                conn, addr = s.accept()
                with conn:
                    while True:
                        try:
                            data = conn.recv(100)
                            # catch graceful disconnection
                            if not data:
                                print(f"Client {addr} disconnected gracefully.")
                                break
                            
                            data = data.decode('utf-8')
                        
                            # decode the string and remove the padding
                            data = data.replace('#', '')
                            state = GameState(data)
                            state.normalize()
                            
                            # feed to lstm
                            lstm_input = torch.tensor(state.feats, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                            player_inputs, hidden, memory = model(lstm_input, hidden, memory, act_last_layer=True)
                            player_inputs = player_inputs.reshape(-1)

                            # send through the socket as a comma separated list of numbers
                            conn.send(bytes(format_pred(player_inputs, threshold) + '\r\n', "utf-8"))

                        except (ConnectionResetError, BrokenPipeError) as e:
                            # catch abrupt network cut
                            print(f"Connection with {addr} was interrupted abruptly: {e}")
                            break
            except(TimeoutError):
                print('TIMEOUTTTTT')
                break