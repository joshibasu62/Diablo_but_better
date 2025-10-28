import gymnasium as gym
import time
from diablo_env.env.diablo_env import DiabloEnv

env = DiabloEnv(render_mode="human")

obs, info = env.reset()

s_size = env.observation_space.shape[0]
a_size = env.action_space.shape[0]

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample())

print("\n_____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample())

try:
    step_count = 0
    max_steps = 1000
    while True: 
        action = env.action_space.sample()  
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        if env.render_mode == "human":
            time.sleep(1.0 / 60.0) 

        if terminated or truncated or step_count >= max_steps:
            obs, info = env.reset()
            step_count = 0

except KeyboardInterrupt:
    print("Simulation stopped by user")

finally:
    env.close() 
