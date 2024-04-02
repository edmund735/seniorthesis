from gymnasium.envs.registration import register

register(
    id='Trading-v0',
    entry_point='trading_env.envs:TradingEnv',
)