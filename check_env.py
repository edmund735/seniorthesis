# import gymnasium
# from gymnasium.utils.env_checker import checkenv
from trading_env import TradingEnv
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import RescaleAction

env = RescaleAction(TradingEnv(), min_action = -1., max_action = 1.)
print(env.action_space)
# It will check your custom environment and output additional warnings if needed
check_env(env)

# check_env(TradingEnv)