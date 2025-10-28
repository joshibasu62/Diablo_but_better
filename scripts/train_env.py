import gymnasium as gym
import numpy as np
import time
from diablo_env.env.diablo_env import DiabloEnv

def train(num_episodes=1000, max_steps=1000, render=False):
    env = DiabloEnv(render_mode="human" if render else None)

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            # Sample random action
            action = env.action_space.sample()

            # Optionally scale down torque to prevent flips
            action = np.clip(action, -5.0, 5.0)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Render
            if render:
                time.sleep(1.0 / 60.0)  # ~60 FPS

            # Reset if done
            if terminated or truncated:
                obs, info = env.reset()

        print(f"Episode {episode+1}/{num_episodes} finished. Total reward: {episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    train(num_episodes=100, max_steps=500)
