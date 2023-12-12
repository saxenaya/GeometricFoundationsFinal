import gymnasium as gym
from stable_baselines3 import DQN
import random
import torch
from typing import List
from copy import deepcopy
import math
from tqdm import tqdm

EVALUATE = True

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
        "order": "sorted",
        "high_speed_reward": 0.5
    }
})

print("Loading model...")
model = DQN.load("custom-roundabout-v0/model")

print("Starting simulation...")
actions = set(range(5))

NUM_SIMULATIONS = 300
num_crashes = 0
total_time = 0
cum_rewards = []
for _ in tqdm(range(NUM_SIMULATIONS)):
    done = truncated = False
    obs, info = env.reset()
    cumulative_reward = 0
    while not (done or truncated):
        # action, _states = model.predict(obs, deterministic=False)
        q_values = model.q_net(torch.from_numpy(obs).unsqueeze(0)).detach().numpy()[0]

        action = q_values.argmax().reshape(-1)
        num_crashes += env.vehicle.crashed
        obs, reward, done, truncated, info = env.step(action)
        cumulative_reward += reward
        if not EVALUATE:
            env.render()
    total_time += env.time
    cum_rewards.append(cumulative_reward)
print("Average reward:", sum(cum_rewards) / len(cum_rewards))   
print("Num crashes:", num_crashes)
print("Total time:", total_time)
