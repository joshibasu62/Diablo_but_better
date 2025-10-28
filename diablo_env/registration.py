from gymnasium.envs.registration import register

register(
    id="DiabloEnv-v0",
    entry_point="diablo_env.env.diablo_env:DiabloEnv",
)
