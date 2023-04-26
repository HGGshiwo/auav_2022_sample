#!/usr/bin/env python
import gym_auav
import gymnasium
import random
import numpy as np

env = gymnasium.make('gym_auav/TrialWorld-v0', render_mode="gui")

seed = 1
random.seed(seed)
np.random.seed(seed)
done = False
for i in range(10):
    obs, info = env.reset(seed=seed)
    while not done:
        action = 1
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated