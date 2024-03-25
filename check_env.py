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


# from stable_baselines3 import PPO
# from trading_env import TradingEnv
# import numpy as np
# import os

# params = {
#     'T': 100.,
#     'impact_decay': 10.,
#     'lamb': 0.01,
#     'gamma': 1,
#     'sigma': 0.3,
#     'c': 0.5,
#     'target_q': 1000,
#     'S0': 10,
# }

# env = TradingEnv(**params)
# print(env.action_space)
# print(env.observation_space)

# model = PPO('MlpPolicy', env, gamma = 1, verbose=2, seed = 29)
# model.learn(1, progress_bar = True)


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from trading_env import TradingEnv
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
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

vec_env = make_vec_env(TradingEnv, n_envs = 1, seed = 29, env_kwargs = params)
vec_env = VecNormalize(vec_env, norm_obs = True, norm_reward = True, gamma = 1)
vec_env.reset()
print(vec_env.step([[0.2]]))
# eval_callback = EvalCallback(vec_env, eval_freq=1,n_eval_episodes=10, deterministic=True,
#                               render=False)
# model = PPO('MlpPolicy', vec_env, gamma = 1, verbose=2)
# model.learn(10, progress_bar = True, callback = eval_callback)

