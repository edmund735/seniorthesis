# import gymnasium
from gymnasium import gymnasium.utils.env_checker.check_env
from trading_env import TradingEnv

check_env(TradingEnv)