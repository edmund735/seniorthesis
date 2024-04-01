from gymnasium.envs.registration import register

register(
     id="TradingEnv",
     entry_point="trading_env.envs:TradingEnv",
)