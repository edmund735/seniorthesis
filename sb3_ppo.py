from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from trading_env import TradingEnv
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import os

params = {
    'init_q': 1000,
    'target_q': 0,
    'c': 0.5,
    'lamb': 0.01,
    'tau': 10,
    'theta': 5,
    'gamma': 0.02,
    'T': 100,
    'sigma': 0.1,
}

vec_env = make_vec_env(TradingEnv, n_envs = 1, seed = 29, env_kwargs = params)
vec_env = VecNormalize(vec_env, norm_obs = False, norm_reward = True, gamma = 1)

# Separate evaluation env, with different parameters passed via env_kwargs
# Eval environments can be vectorized to speed up evaluation.
eval_env = make_vec_env(TradingEnv, n_envs = 1, seed = 11, env_kwargs = params)
eval_env = VecNormalize(eval_env, norm_obs = False, norm_reward = True, gamma = 1)
# Create callback that evaluates agent for 5 episodes every 500 training environment steps.
# When using multiple training environments, agent will be evaluated every
# eval_freq calls to train_env.step(), thus it will be evaluated every
# (eval_freq * n_envs) training steps. See EvalCallback doc for more information.

# Create log dir where evaluation results will be saved
eval_log_dir = "./eval_logs/"
os.makedirs(eval_log_dir, exist_ok=True)

eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                              log_path=eval_log_dir, eval_freq=100,
                              n_eval_episodes=10, deterministic=True,
                              render=False)

model = PPO('MlpPolicy', vec_env, gamma = 1, verbose=2, seed = 30)
model.learn(10_000, progress_bar = True, callback = eval_callback)
model.save('ppo_test')

# env = TradingEnv(**params)







# ### Evaluation
# results = evaluate_policy(model,
#                 env,
#                 n_eval_episodes = 10_000,
#                 deterministic=True,
#                 reward_threshold = 100,
#                 return_episode_rewards=True,
#                 warn = True)
# print(results)




# model = PPO("MlpPolicy", vec_env, learning_rate = 0.001, clip_range = 0.3, verbose=2)
# model.learn(total_timesteps=100_000, progress_bar = True)
# model.save("ppo_test")

# # model = PPO.load("ppo_test")

# obs = vec_env.reset()

# cum_rew = np.zeros(1000)
# rew = np.zeros(1000)

# for i in range(100):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     # print(f"i={i}, action={action[1]}, state={_states[1]}, reward={rewards[1]}")
#     rew = rewards[0]
#     if i == 0:
#         cum_rew[i] = rew
#     else:
#         cum_rew[i] = cum_rew[i-1] + rew
#     print(f"{action[0][0]},")
# print('a')
# for j in range(100):
#     print(f"{cum_rew[j]},")
