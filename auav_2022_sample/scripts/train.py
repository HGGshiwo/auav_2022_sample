#!/usr/bin/env python3
from TrialEnv import TrialEnv
import numpy as np

if __name__ == "__main__":
    env = TrialEnv()
    total_num_episodes = 10
    for i in range(total_num_episodes):
        done = False
        obs, info = env.reset()
        while not done:
            action = np.random.randint(0, 7)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
