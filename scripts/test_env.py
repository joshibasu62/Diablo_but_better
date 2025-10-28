import gymnasium as gym
import time
from diablo_env.env.diablo_env import DiabloEnv

env = DiabloEnv(render_mode="human")

obs, info = env.reset()

s_size = env.observation_space.shape[0]
a_size = env.action_space.shape[0]

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action

for _ in range(1000):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)

    if env.render_mode == "human":
        time.sleep(1.0 / 60.0)  # ~60 FPS

    if terminated or truncated:
        obs, info = env.reset()

env.close()
