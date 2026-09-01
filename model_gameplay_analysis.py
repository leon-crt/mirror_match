import SF3_environment as SF3Env
from SF3_environment.wrappers import FlattenObservation
import torch
from model import LSTM
from util import format_pred_env, normalize
import gymnasium
from rl_util import transpose_weights_nn_to_rl

# TODO:
#       - check for super usage (during one frame there was x amount of gauge and the next is decreased a lot)

# params
max_rounds = 2

# model init
in_size = 36
hidden_size = 512
out_size = 10
num_layers = 2
model = LSTM(input_size=in_size, output_size=out_size, num_layers=num_layers, hidden_size=hidden_size)
hidden = torch.zeros([num_layers, 1, hidden_size])
memory = torch.zeros([num_layers, 1, hidden_size])
threshold = 0.5
RL_weights = False
autoreg = True

device = torch.device("cpu")
ch_path = 'checkpoints/checkpoint_249'
checkpoint = torch.load(ch_path, map_location=device)
if RL_weights:
    model = transpose_weights_nn_to_rl(checkpoint, model)
else:
    model.load_state_dict(checkpoint['model_state_dict'])

# environment init
base_env = gymnasium.make("SF3_environment/StreetFighter3-v0", render_mode="human", mode="test")
env = FlattenObservation(base_env)

state, info = env.reset()
log_rewards = []
done = False
round_number = 0
out = torch.zeros((10,))

# logger variables
log = {
    "states": [state],
    "model_action_code": [info["modelAction"]],
    "model_action_group": [info["modelActionGroup"]],
    "done" : [False]
    }

while(round_number < max_rounds):
    state = normalize(state)
    state = torch.tensor(state, dtype=torch.float32)
    if autoreg:
        state = torch.concat((state, out))
    out, hidden, memory = model(state.unsqueeze(0).unsqueeze(0), hidden, memory, act_last_layer=True)
    out = out.reshape(-1)
    action = format_pred_env(out, threshold)
    state, reward, done, _, info = env.step(action) 
    # log stuff
    log["states"].append(state.copy())
    log["model_action_code"].append(info["modelAction"])
    log["model_action_group"].append(info["modelActionGroup"])
    log["done"].append(done)

    log_rewards.append(reward)
    if done:
        round_number += 1

env.close()  

# init count variables
antiair_count = 0
throw_tech_count = 0
ground_tech_count = 0
parry_count = 0
max_combo_len = 0
performed_combos_count = 0
combo_count = 0

# init status variables
hit_opp = False
jumping_opp = False
is_parrying = False
is_ground_teching = False
is_throw_teching = False

# when done analyze states
for frame in range(len(log["done"]) - 1):

    prev_state_model = log["states"][frame][:8]
    state_model = log["states"][frame + 1][:8]
    prev_state_opp = log["states"][frame][8:16]
    state_opp = log["states"][frame + 1][8:16]
    group_action_prev = log["model_action_group"][frame]
    group_action = log["model_action_group"][frame + 1]
    action_prev = log["model_action_code"][frame]
    action = log["model_action_code"][frame + 1]

    hit_opp_prev = prev_state_opp[6]
    hit_opp = state_opp[6]

    # if the next is the last frame of the round
    if log["done"][frame+1]:
        if combo_count > 0:
            if max_combo_len < combo_count:
                max_combo_len = combo_count
                combo_count = 0
                performed_combos_count += 1
        is_parrying = False
        is_ground_teching = False
        is_throw_teching = False
        jumping_opp = False
        continue

    # Jumping state management and antiair detection
    if not jumping_opp and state_opp[1] > 30 and not hit_opp:
        jumping_opp = True
    if jumping_opp:
        if hit_opp:
            antiair_count += 1
            jumping_opp = False
        elif state_opp[1] <= 10:
            jumping_opp = False

    # check for combo length by checking if state changes between 4 and 5 while opponent is hit
    if hit_opp:
        if hit_opp_prev and action_prev != action and (group_action == 4 or group_action == 5):
            combo_count += 1

    # if opponent was previously in hitstun, in a combo, but now they are not (when running combo is over), update the combo related vars
    if hit_opp_prev and (combo_count > 0) and (not hit_opp):
        if max_combo_len < combo_count:
            max_combo_len = combo_count
        combo_count = 0
        performed_combos_count += 1

    # check for parries
    if not is_parrying and (action in range(24,28) and group_action == 0):
        is_parrying = True
        parry_count += 1
    elif is_parrying and ((action not in range(24,28)) or (group_action != 0)):
        is_parrying = False

    # check for ground techs
    if not is_ground_teching and (action == 71 and group_action == 1):
            is_ground_teching = True
            ground_tech_count += 1
    elif is_ground_teching and ((action not in range(71,73)) or (group_action != 1)):
        is_ground_teching = False

    # check for throw techs
    if not is_ground_teching and (action == 43 and group_action == 0):
            is_throw_teching = True
            throw_tech_count += 1
    elif is_ground_teching and ((action != 43) or (group_action != 0)):
        is_throw_teching = False

print(f"Number of performed antiairs: {antiair_count}")
print(f"Number of performed throw techs: {throw_tech_count}")
print(f"Number of performed ground techs: {ground_tech_count}")
print(f"Number of performed parries: {parry_count}")
print(f"Number of max combo hits: {max_combo_len}")
print(f"Number of combos performed: {performed_combos_count}")