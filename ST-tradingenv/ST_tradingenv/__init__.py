from gymnasium.envs.registration import register

register(
    id="ST_tradingenv/Trading-v0",
    entry_point="ST_tradingenv.envs:TradingEnv",
)
