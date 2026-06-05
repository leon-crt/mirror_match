import os
import subprocess
import socket
import time
import win32process
import win32gui
import win32api
import win32con
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime

from util import GameState, format_pred
from model import LSTM

HOST = "127.0.0.1"
PORT = 42069

# TODO:
#       - emulator crashes

# lstm initialization
device = torch.device(0 if torch.cuda.is_available() else 'cpu')
hidden_size = 128
out_size = 12 # number of pressable buttons same as targets
match_lstm = LSTM(input_size=28, output_size=out_size, hidden_size=hidden_size).to(device)
hidden = torch.zeros(1, 1, hidden_size, device=device)
memory = torch.zeros(1, 1, hidden_size, device=device)

# load model
ch_path = 'checkpoints/checkpoint_final'
checkpoint = torch.load(ch_path, map_location=device)
match_lstm.load_state_dict(checkpoint['model_state_dict'])

# test LSTM
# state = GameState('424,0,161,24,0,0,0,0,600,0,161,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
# state.normalize()
# lstm_input = torch.tensor(state.feats, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# start emulator
emu_proc = subprocess.Popen(["./fbneo/fcadefbneo.exe", "sfiii3nr1", "./model_game_interface.lua"])

output_vec = []
time_vec = []
# establish tcp connection
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST,PORT))
    s.settimeout(2)
    try:
        # game loop
        while True:
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
                        elab_time_st = datetime.datetime.now()
                        # decode the string and remove the padding
                        data = data.decode('utf-8').replace('#', '')
                        state = GameState(data)
                        state.normalize()
                        lstm_input = torch.tensor(state.feats, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        # feed to lstm
                        player_inputs, hidden, memory = match_lstm(lstm_input, hidden, memory)
                        player_inputs = player_inputs.squeeze(0).squeeze(0)
                        output_vec.append(format_pred(player_inputs))
                        # send through the socket as a comma separated list of numbers
                        conn.send(bytes(format_pred(player_inputs) + '\r\n', "utf-8"))
                        # if even:
                        #     conn.send(bytes('0,0,0,0,0,0,0,1,0,0,0,1\r\n', "utf-8"))
                        # else:
                        #     conn.send(bytes('0,0,0,0,0,0,0,0,0,0,0,0\r\n', "utf-8"))
                        # even = not even
                        elab_time_end = datetime.datetime.now() - elab_time_st
                        time_vec.append(elab_time_end)

                    except (ConnectionResetError, BrokenPipeError) as e:
                        # catch abrupt network cut
                        print(f"Connection with {addr} was interrupted abruptly: {e}")
                        break
    except(TimeoutError):
        time_vec = np.array(time_vec)
        print(time_vec.mean())
        # print(f'max time: {time_vec.max()}' )
        # print(f'min time: {time_vec.min()}' )
        # print(f'avg time: {time_vec.mean()}' )
        # output_vec = np.array(output_vec)
        # for i in range(len(output_vec)-1):
        #     plt.hist(output_vec[:,i])
        #     plt.title(str(i))
        #     plt.show()



