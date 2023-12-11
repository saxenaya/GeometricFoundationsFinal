import gymnasium as gym
from stable_baselines3 import DQN
import random
import torch
from typing import List
from copy import deepcopy
import math

print("Creating Environment...")
env = gym.make("roundabout-v0", render_mode="rgb_array")
env.configure({
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 6,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted"
    }
})

print("Loading model...")
model = DQN.load("roundabout-v0/model")

actions_dict = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

def get_leading_car(obs):
    _, player_x, player_y, player_vx, player_vy = obs[0]
    player_heading = math.atan2(player_vy, player_vx)
    best_heading_diff, leading_car = 1e99, None
    for other_car in obs[1:]:
        if other_car[0] == 0:
            continue
        _, other_x, other_y, _, _ = other_car
        other_heading = math.atan2(other_y - player_y, other_x - player_x)
        heading_diff = abs(player_heading - other_heading)
        if heading_diff > (math.pi / 4):
            continue
        if heading_diff < best_heading_diff:
            best_heading_diff = heading_diff
            leading_car = other_car
    return leading_car


def get_neighboring_states(env: gym.Env, actions: set) -> List[List[float]]:
    # Get observations
    # TODO: This is slow and it lags the rendering. Is there a better way to determine the observations
    # after applying each action to a state?
    return [deepcopy(env).step(action) for action in actions]


def is_unsafe(observation, info):
    if info['crashed']:
        print(f"Taking action is unsafe! Expected collision!")
        return True
    # print("Observation:", observation)
    # print("Leading car:", get_leading_car(observation))
    _, player_x, player_y, player_vx, player_vy = observation[0]
    leading_car = get_leading_car(observation)
    if leading_car is None:
        return False
    
    _, lc_x, lc_y, lc_vx, lc_vy = leading_car
    
    player_velocity = (player_vx ** 2 + player_vy ** 2) ** 0.5
    lc_velocity = (lc_vx ** 2 + lc_vy ** 2) ** 0.5

    p_safe = (player_velocity - lc_velocity) ** 2
    distance_to_lc = ((player_x - lc_x) ** 2 + (player_y - lc_y) ** 2) ** 0.5

    if distance_to_lc < p_safe:
        print(f"Taking action is unsafe! p_safe violation!")

    return distance_to_lc < p_safe


def robust_q_masking(q_values, actions):
    neighbor_states = get_neighboring_states(env, actions)
    neighbor_observations = [st[0] for st in neighbor_states]
    neighbor_infos = [st[-1] for st in neighbor_states]
    
    for idx, action in enumerate(actions):
        # Each observation is a V x F array that describes a list of V nearby vehicles using
        # F features: presence, x, y, vx, vy.
        # More information can be found here: https://highway-env.farama.org/observations/#kinematics
        observation = neighbor_observations[idx]
        info = neighbor_infos[idx]

        if is_unsafe(observation, info):
            print(f"Action {actions_dict[action]} is unsafe!")
            q_values[idx] = 0
    
    return q_values


print("Starting simulation...")
actions = set(range(5))
while True:
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        # action, _states = model.predict(obs, deterministic=False)
        q_values = model.q_net(torch.from_numpy(obs).unsqueeze(0)).detach().numpy()[0]

        q_values = robust_q_masking(q_values, actions)

        action = q_values.argmax().reshape(-1)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
