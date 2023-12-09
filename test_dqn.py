import gymnasium as gym
from stable_baselines3 import DQN
import random
import torch
from typing import List
from copy import deepcopy

print("Creating Environment...")
env = gym.make("roundabout-v0", render_mode="rgb_array")
env.configure({
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 3,
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


def get_neighboring_observations(env: gym.Env, actions: set) -> List[List[float]]:
    # Get observations
    # TODO: This is slow and it lags the rendering. Is there a better way to determine the observations
    # after applying each action to a state?
    return [deepcopy(env).step(action)[0] for action in actions]


def is_unsafe(observation):
    # TODO: Implement
    return False


def robust_q_masking(q_values, actions):
    neighbor_observations = get_neighboring_observations(env, actions)
    # print("Neighbor observations:", neighbor_observations)
    # TODO: Loop through observations of neighboring states and mask out Q-values of unsafe states.
    for idx, (action, new_observation) in enumerate(zip(actions, neighbor_observations)):
        # Each observation is a V x F array that describes a list of V nearby vehicles using
        # F features: presence, x, y, vx, vy.
        # More information can be found here: https://highway-env.farama.org/observations/#kinematics
        # print("Action:", action)
        # print("New observation:", new_observation)
        q_values[idx] = 0 if is_unsafe(new_observation) else q_values[idx]
    
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
