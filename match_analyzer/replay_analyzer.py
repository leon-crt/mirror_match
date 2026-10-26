from handler import SF3ReplayExtractor
import numpy as np
import matplotlib.pyplot as plt

replay_file = "C:/Users/leonp/Games/Fightcade/emulator/fbneo/recordings/virsaga-jme.fr"

env = SF3ReplayExtractor(replay_file, "human")

state, info = env.reset()
log_rewards = []
done = False
round_number = 0
max_rounds = 2

# logger variables
log = {
    "states": [state],
    "p1_action_code": [info["modelAction"]],
    "p1_action_group": [info["modelActionGroup"]],
    "done" : [False]
    }

# init count variables
antiair_count = 0
throw_tech_count = 0
ground_tech_count = 0
parry_count = 0
max_combo_len = 0
performed_combos_count = 0
combo_count = 0
player_jumps = 0
player_meter = 0
prev_player_meter = 0
supers_performed = 0
throws_performed = 0

# init status variables
hit_opp = False
hit_pl = False
jumping_opp = False
jumping_pl = False
is_parrying = False
is_ground_teching = False
is_throw_teching = False

done = False
starting_state = [424, 0, 161, 0, 0, 0, 0, 0, 600, 0, 161, 0, 0, 0, 0, 0] + ([0] * 10)
state = []
while(round_number < max_rounds):
        if len(state) < 1:
            prev_state_model = starting_state[:8]
            prev_state_opp = starting_state[8:16]
            group_action_prev = 0
            action_prev = 0
        else:
            prev_state_model = state_model
            prev_state_opp = state_opp
            group_action_prev = group_action
            action_prev = action

        state, reward, done, info = env.step()
        # log stuff
        log["states"].append(state)
        log["p1_action_code"].append(info["modelAction"])
        log["p1_action_group"].append(info["modelActionGroup"])
        log["done"].append(done)
        log_rewards.append(reward)

        state_model = state[:8]
        state_opp = state[8:16]
        group_action = info["modelActionGroup"]
        action = info["modelAction"]
    
        hit_opp_prev = bool(prev_state_opp[6])
        hit_opp = bool(state_opp[6])
        hit_pl = bool(state_model[6])

        prev_player_meter = prev_state_model[3]
        player_meter = state_model[3]

        if done:
            state = []
            round_number += 1
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
        if jumping_opp and (not jumping_pl):
            if hit_opp:
                antiair_count += 1
                jumping_opp = False
        if jumping_pl and state_opp[1] <= 10:
            jumping_opp = False

        if not jumping_pl and state_model[1] > 30 and not hit_pl:
            jumping_pl = True
            player_jumps += 1
        if jumping_pl and state_model[1] <= 10:
            jumping_pl = False

        # Super usage detection
        if prev_player_meter > player_meter + 65:
            supers_performed += 1
    
        # check for combo length by checking if state changes between 4 and 5 while opponent is hit
        if hit_opp:
            if hit_opp_prev and action_prev != action and (group_action == 4 or group_action == 5):
                combo_count += 1
    
        # if opponent was previously in hitstun, in a combo, but now they are not (when running combo is over), update the combo related vars
        if hit_opp_prev and (combo_count > 1) and (not hit_opp):
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
        if not is_throw_teching and (action == 43 and group_action == 0):
                is_throw_teching = True
                throw_tech_count += 1
        elif is_throw_teching and ((action != 43) or (group_action != 0)):
            is_throw_teching = False

        # check for successful grabs
        if group_action == 2 and group_action_prev != 2:
            throws_performed += 1

env.close()

print(f"Number of performed antiairs: {antiair_count}")
print(f"Number of performed throw techs: {throw_tech_count}")
print(f"Number of performed ground techs: {ground_tech_count}")
print(f"Number of performed parries: {parry_count}")
print(f"Number of max combo hits: {max_combo_len}")
print(f"Number of combos performed: {performed_combos_count}")
print(f"Number of jumps: {player_jumps}")
print(f"Number of performed supers: {supers_performed}")
print(f"Number of successful throws: {throws_performed}")