import gymnasium as gym
from stable_baselines3 import DQN
import random
import torch

print("Creating Environment...")
env = gym.make("roundabout-v0", render_mode="human")

print("Loading model...")
model = DQN.load("roundabout-v0/model")

print("Starting simulation...")
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    # action, _states = model.predict(obs, deterministic=False)
    q_values = model.q_net(torch.from_numpy(obs).unsqueeze(0))
    action = q_values.argmax(dim=1).reshape(-1)
    obs, reward, done, truncated, info = env.step(action)
    env.render()