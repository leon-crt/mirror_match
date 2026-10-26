import subprocess
import socket
import win32process
import win32gui
import win32api
import win32con
import numpy as np
import gymnasium as gym
import psutil
import os
from pathlib import Path
import sys

HOST = "127.0.0.1"
base_port = 50000

input_size = 10

# TODO:

emu_path = "fbneo/fcadefbneo.exe"

def enumWindowsProc(hwnd, lParam):
    if (lParam is None) or ((lParam is not None) and win32process.GetWindowThreadProcessId(hwnd)[1] == lParam):
        text = win32gui.GetWindowText(hwnd)
        if text:
            win32api.SendMessage(hwnd, win32con.WM_CLOSE)

class SF3ReplayExtractor(gym.Env):
    metadata = {"render_modes": ["human", "turbo"], "render_fps": 60}

    def __init__(self, replay_file, render_mode="human"):
        self.replay_file = replay_file
        self.min_state_limits = np.array([93, -42, 0, 0, 0, 0, 0, 0])
        self.max_state_limits = np.array([928, 226, 161, 336, 70, 1, 1, 1])
        self.observation_space = gym.spaces.Dict({
            'player_state': gym.spaces.Box(self.min_state_limits, self.max_state_limits, (8,), np.float32),
            'opponent_state': gym.spaces.Box(self.min_state_limits, self.max_state_limits, (8,), np.float32),
            'opponent_inputs': gym.spaces.Box(0,1,(input_size,), np.int8),
        })
        self.action_space = gym.spaces.MultiBinary(input_size)
        self._player_state = np.array([-100] * 8)
        self._opp_state = np.array([-100] * 8)
        self._opp_inputs = np.array([-1] * input_size)
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.socket = None
    
    def _get_used_ports(self):
        """Returns a set of all TCP ports currently bound/in-use by the OS."""
        used_ports = set()
        # Retrieve all active network connections from the OS kernel
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr:
                used_ports.add(conn.laddr.port)
        return used_ports
    
    def _is_port_free(self, port: int) -> bool:
        """Checks if a port is unlisted in the OS connection table."""
        return port not in self._get_used_ports()
    
    def _get_obs(self):
        return {
            'player_state': self._player_state,
            'opponent_state': self._opp_state,
            'opponent_inputs': self._opp_inputs
        }
    
    def flatten_obs(self, obs):
        return  np.concat([obs["player_state"], obs['opponent_state'], obs['opponent_inputs']])
    
    def _parse_state(self, state):
        res = state.split(sep=',')[:-1]
        res = res[:-4] + res[-2:]

        res = [int(x) for x in res]
        return res
    
    def _update_fields(self, state):
        self._player_state = np.array(state[:len(self._player_state)], dtype=np.float32)
        self._opp_state = np.array(state[len(self._player_state):-len(self._opp_inputs)], dtype=np.float32)
        self._opp_inputs = np.array(state[-len(self._opp_inputs):], dtype=np.int8)
    
    def reward(self, prev_obs, new_obs):
        r = 0
        health_pl = 2
        health_opp = 10
        meter_pl = 3
        stun_pl = 5
        stun_opp = 13
        is_hit_pl = 6
        is_hit_opp = 14
        is_thrown_pl = 7
        is_thrown_opp = 15

        # get arrays from the observation dictionaries
        prev_state = self.flatten_obs(prev_obs)
        new_state = self.flatten_obs(new_obs)

        # check health values
        if new_state[health_pl] == 0: # terminal failure (player lost)
            r = -10
        elif new_state[health_opp] == 0: # terminal success (opponent lost)
            r = 10
        if (new_state[is_hit_pl] and (not prev_state[is_hit_pl])) or (new_state[is_thrown_pl] and (not prev_state[is_thrown_pl])): # player got hit
            r -= 0.35
        if (new_state[is_hit_opp] and (not prev_state[is_hit_opp])) or (new_state[is_thrown_opp] and (not prev_state[is_thrown_opp])): # opponent got hit or thrown
            r += 0.3
        if new_state[is_hit_opp]:
            r += 0.05
        # time-step penalty
        if new_state[health_pl] <= new_state[health_opp]:
            r -= 0.05
        else:
            r -= 0.01
        
        # stun related rewards and penalties
        if new_state[stun_pl] == 1 and prev_state[stun_pl] == 0:
            r -= 0.5
        if new_state[stun_opp] == 1 and prev_state[stun_opp] == 0:
            r += 0.5
        
        return r
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info = {}
        
        # check if emulator instance is already running and close it in case it is
        if self.window != None or self.socket != None:
            self.close()

        # TODO: if addess already in use do port + 1 and same for lua
        num_attempts = 0
        connection_succeeded = False

        free_port = None
        curr_port = base_port
        while free_port == None:
            if self._is_port_free(curr_port):
                free_port = curr_port
            else:
                curr_port += 1

        # add port to environment for child processes so that lua side can read it
        env = os.environ.copy()
        env["EMU_PORT"] = str(free_port)

        while not connection_succeeded:
            # start emulator
            self.window = subprocess.Popen([str(emu_path), self.replay_file, "./event_extractor.lua"], env=env)

            # establish tcp connection
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((HOST, free_port))
            try:
                # send configuration
                config = f"{self.render_mode}"
                self.socket.send(bytes(config + '\r\n', "utf-8"))

                # receive first state
                data = self.socket.recv(100)
                # catch graceful disconnection
                if not data:
                    raise ConnectionError(f"Client {HOST} disconnected gracefully.")
                
                data = data.decode('utf-8')
            
                # decode the string and remove the padding
                data = data.replace('#', '')
                state = self._parse_state(data)
                info["modelAction"] = state[-2]
                info["modelActionGroup"] = state[-1]
                state = state[:26]

                # update class variables
                self._update_fields(state)
                connection_succeeded = True
                
            except (ConnectionResetError, BrokenPipeError) as e:
                # catch abrupt network cut
                num_attempts += 1
                self.close()
                if num_attempts >= 3:
                    raise ConnectionError(f"Connection with {HOST} was interrupted abruptly: {e}")

        return self.flatten_obs(self._get_obs()), info

    # Step function for when mode=selfplay so that inputs for both players are provided
    def step(self):
        info = {}
        data = "continue"
        self.socket.send(bytes(data + '\r\n', "utf-8"))

        terminated = False

        # receive next game state
        data = self.socket.recv(100)

        # catch graceful disconnection
        if not data:
            raise ConnectionError(f"Client {HOST} disconnected gracefully.")
        
        data = data.decode('utf-8')
        data = data.replace('#', '')
        # terminal state
        if data[-1] == 'R':
            data = self._parse_state(data[:-1])
            terminated = True
        else:
            data = self._parse_state(data)

        info["modelAction"] = data[-2]
        info["modelActionGroup"] = data[-1]
        data = data[:26]
        
        previous_state = self._get_obs()
        # update class variables
        self._update_fields(data)
        observation = self._get_obs()

        # compute reward
        r = self.reward(previous_state, observation)

        return self.flatten_obs(observation), r, terminated, info

    def close(self):
        if self.socket != None:
            self.socket.close()
            self.socket = None
        if self.window != None:
            win32gui.EnumWindows(enumWindowsProc, self.window.pid)