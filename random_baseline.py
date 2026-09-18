import SF3_environment as SF3Env
from SF3_environment.wrappers import FlattenObservation
import torch
import matplotlib.pyplot as plt
import numpy as np
import gymnasium

base_env = gymnasium.make("SF3_environment/StreetFighter3-v0", render_mode="human", mode="free")
env = FlattenObservation(base_env)

state, _ = env.reset()
log_rewards = []
done = False

while(True):
    seed = torch.empty(10,).uniform_(0,1)
    action = torch.bernoulli(seed).numpy()
    state, reward, done, _, _ = env.step(action) 
