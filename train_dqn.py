import gymnasium as gym
import highway_env
from stable_baselines3 import DQN

env = gym.make("custom-roundabout-v0")
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
        "high_speed_reward": 0.5,
    }
})
env.reset()
model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="roundabout-v0/")
model.learn(int(2e4))
model.save("roundabout-v0/model")