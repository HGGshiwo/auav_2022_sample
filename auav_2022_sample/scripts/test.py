#!/usr/bin/env python3
from Env import Env
from A2C import A2C
import torch
import numpy as np

if __name__ == "__main__":
    pre_train = "20230502172814.pt"
    use_rl = False  # 测试时是否使用强化学习辅助
    use_cuda = False

    env = Env(use_odom=True)
    obs_shape = env.obs.shape
    action_shape = len(env.actions)

    # set the device
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    agent = A2C(
        obs_shape,
        action_shape,
        device,
        critic_lr=1,
        actor_lr=1,
        random_rate=0,
    )
    agent.eval()
    states, info = env.reset()
    rewards = []
    for i in range(100):
        action = 0
        if use_rl:
            actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                states
            )
            action = actions.cpu().numpy()

        states, reward, terminated, truncated, infos = env.step(action)
        rewards.append(reward)

    env.log(f"Test end with average reward: {np.array(rewards).mean()}")
