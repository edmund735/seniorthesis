from gymnasium.envs.registration import register

register(
     id="trading_env/TradingEnv",
     entry_point="trading_env.envs:TradingEnv",
     max_episode_steps=100,
)