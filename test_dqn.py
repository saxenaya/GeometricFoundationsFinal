import gymnasium as gym
from stable_baselines3 import DQN
import random
import torch
from typing import List
from copy import deepcopy

print("Creating Environment...")
env = gym.make("roundabout-v0", render_mode="rgb_array")

print("Loading model...")
model = DQN.load("roundabout-v0/model")

def get_neighboring_observations(env : gym.Env, actions : set) -> List[List[float]]:
  # Get observations
  # TODO: This is slow and it lags the rendering. Is there a better way to determine the observations
  # after applying each action to a state?
  return [deepcopy(env).step(action)[0] for action in actions]

def robust_q_masking(q_values, actions):
  neighbor_observations = get_neighboring_observations(env, actions)
  # TODO: Loop through observations of neighboring states and mask out Q-values of unsafe states.
  return q_values

print("Starting simulation...")
actions = set(range(5))
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    # action, _states = model.predict(obs, deterministic=False)
    q_values = model.q_net(torch.from_numpy(obs).unsqueeze(0))

    q_values = robust_q_masking(q_values, actions)

    action = q_values.argmax(dim=1).reshape(-1)
    obs, reward, done, truncated, info = env.step(action)
    env.render()