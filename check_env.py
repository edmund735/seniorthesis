from trading_env import TradingEnv

# env = TradingEnv()
# obs, info = env.reset()
# n_steps = 10
# for i in range(n_steps):
#     # Random action
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(f"\ni={i}")
#     print(f"action: {action}")
#     print(f"observation: {obs}")
#     print(f"reward: {reward}")
#     if terminated:
#         obs, info = env.reset()


from stable_baselines3 import PPO
from trading_env import TradingEnv
import numpy as np
import os

params = {
    'T': 100.,
    'impact_decay': 10.,
    'lamb': 0.01,
    'gamma': 1,
    'sigma': 0.3,
    'c': 0.5,
    'target_q': 1000,
    'S0': 10,
}

env = TradingEnv(**params)
print(env.action_space)
print(env.observation_space)

model = PPO('MlpPolicy', env, gamma = 1, verbose=2, seed = 29)
model.learn(1, progress_bar = True)

