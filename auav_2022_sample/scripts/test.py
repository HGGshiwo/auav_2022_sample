#!/usr/bin/env python3
from Env2 import Env
from A2C import A2C
from A3Cv3 import A3C
import torch
import numpy as np

if __name__ == "__main__":
    pre_train = "20230509092114.pt"
    use_cuda = False

    env = Env(use_odom=True)
    obs_shape = env.obs_shape
    action_shape = len(env.actions)

    # set the device
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    agent = A3C(
        obs_shape,
        action_shape,
        device,
        critic_lr=1,
        actor_lr=1,
        random_rate=0,
    )    
    
    if pre_train != None:
        config = torch.load(pre_train)
        agent.load_state_dict(config["best_agent"])
    
    agent.eval()
    states, info = env.reset()
    rewards = []
    for i in range(100):
        action = 0
        if pre_train != None:
            actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                states
            )
            action = actions.cpu().numpy()

        states, reward, terminated, truncated, infos = env.step(action)
        rewards.append(reward)

    env.log(f"Test end with average reward: {np.array(rewards).mean()}")
