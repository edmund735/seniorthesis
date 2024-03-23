import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from trading_env import TradingEnv

# Parallel environments
vec_env = make_vec_env(TradingEnv, n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_test")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_test")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)