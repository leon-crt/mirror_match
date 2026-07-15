import os
import subprocess
import socket
import win32process
import win32gui
import win32api
import win32con
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

from util import GameState, format_pred
from model import LSTM, reward

HOST = "127.0.0.1"
PORT = 42069

# lstm initialization
device = torch.device(0 if torch.cuda.is_available() else 'cpu')
hidden_size = 128
out_size = 12 # number of pressable buttons same as targets
match_lstm = LSTM(input_size=28, output_size=out_size, hidden_size=hidden_size).to(device)
threshold = 0.2

# load model
ch_path = 'checkpoints/checkpoint_273_Harmonaz'
checkpoint = torch.load(ch_path, map_location=device)
match_lstm.load_state_dict(checkpoint['model_state_dict'])

def enumWindowsProc(hwnd, lParam):
    if (lParam is None) or ((lParam is not None) and win32process.GetWindowThreadProcessId(hwnd)[1] == lParam):
        text = win32gui.GetWindowText(hwnd)
        if text:
            win32api.SendMessage(hwnd, win32con.WM_CLOSE)

def get_trajectories_vs_CPU(model, hidden_size, device, round_num, CPU=True):

    hidden = torch.zeros(1, 1, hidden_size, device=device)
    memory = torch.zeros(1, 1, hidden_size, device=device)
    trajectories = []

    while len(trajectories) < round_num:
        # start emulator
        emu_proc = subprocess.Popen(["./fbneo/fcadefbneo.exe", "sfiii3nr1", "./model_game_interface.lua"])

        actions = []
        states = []
        rewards = []
        round_counter = 0
        match_finished = False
        # establish tcp connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST,PORT))
            s.settimeout(10)
            # game loop
            while not match_finished and (len(trajectories) < round_num):
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

                                # if data contains 'R' then the round is over and we store the trajectory before moving to the next
                                if data[-1] == 'R':
                                    round_counter += 1
                                    data = data.replace('#', '')
                                    state = GameState(data[:-1])
                                    state.normalize()
                                    rewards.append(reward(states[-1], state.feats))
                                    states.append(state.feats)
                                    actions.append(actions[-1]) # duplicate last model output so states and actions have the same length
                                    # save the trajectory
                                    trajectories.append([torch.tensor(states), torch.tensor(actions), torch.tensor(rewards)])
                                    states, actions, rewards = [], [], []
                                    # Send the player inputs for the last frame otherwise the lua script throws a tantrum
                                    conn.send(bytes(format_pred(torch.zeros(12), threshold) + '\r\n', "utf-8"))
                                    if round_counter >= 2:
                                        match_finished = True
                                        round_counter = 0
                                    break

                                # decode the string and remove the padding
                                data = data.replace('#', '')
                                state = GameState(data)
                                state.normalize()
                                # calculate the reward and update environment variables
                                if len(states) > 0:
                                    rewards.append(reward(states[-1], state.feats))
                                states.append(state.feats)
                                # feed to lstm
                                lstm_input = torch.tensor(state.feats, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                                player_inputs, hidden, memory = model(lstm_input, hidden, memory, act_last_layer=True)
                                player_inputs = player_inputs.squeeze(0).squeeze(0)
                                actions.append(player_inputs.tolist())
                                # send through the socket as a comma separated list of numbers
                                conn.send(bytes(format_pred(player_inputs, threshold) + '\r\n', "utf-8"))
                            except (ConnectionResetError, BrokenPipeError) as e:
                                # catch abrupt network cut
                                print(f"Connection with {addr} was interrupted abruptly: {e}")
                                break
                except(TimeoutError):
                    print('TIMEOUTTTTT')
                    break
            
        win32gui.EnumWindows(enumWindowsProc, emu_proc.pid)

    return trajectories[:round_num]