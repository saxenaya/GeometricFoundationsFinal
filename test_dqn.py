import gymnasium as gym
import highway_env
from stable_baselines3 import DQN

print("Creating Environment...")
env = gym.make("highway-fast-v0", render_mode="human")

print("Loading model...")
model = DQN.load("highway_dqn/model")

print("Starting simulation...")
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()