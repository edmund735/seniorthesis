import gymnasium as gym
from trading_env_rllib import TradingEnv as TradingEnv1
import trading_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env
import numpy as np
import os

# env = TradingEnv()
env = gym.make('Trading-v0')
gym_check_env(env, skip_render_check = True)
# env = TradingEnv()

obs, info = env.reset()
print(obs)
n_steps = 10
for i in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\ni={i}")
    print(f"action: {action}")
    print(f"observation: {obs}")
    print(f"reward: {reward}")
    if terminated:
        obs, info = env.reset()


# vec_env = make_vec_env('Trading-v0', n_envs = 2, seed = 29)
# vec_env = VecNormalize(vec_env, norm_obs = False, norm_reward = True, gamma = 1)

# obs = vec_env.reset()
# print(obs)
# n_steps = 10
# # x = vec_env.action_space.sample()
# # print(x)
# # print(vec_env.step(x))
# for i in range(n_steps):
#     # Random action
#     action = vec_env.action_space.sample()
#     print(action)
#     obs, reward, done, info = vec_env.step(action)
#     print(f"\ni={i}")
#     print(f"action: {action}")
#     print(f"observation: {obs}")
#     print(f"reward: {reward}")
#     if done:
#         obs, info = vec_env.reset()

# eval_callback = EvalCallback(vec_env, eval_freq=1,n_eval_episodes=10, deterministic=True,
#                               render=False)
# model = PPO('MlpPolicy', vec_env, gamma = 1, verbose=2)
# model.learn(10, progress_bar = True, callback = eval_callback)

