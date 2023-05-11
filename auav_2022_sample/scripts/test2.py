#!/usr/bin/env python3
from Env4 import Env
from A2C import A2C
from A3Cv3 import A3C
import torch
import numpy as np


class Test(Env):
    def __init__(self, pre_train, use_odom=True, use_KF=False, use_cuda=True):
        super().__init__(use_odom, use_KF)
        self.agent_ready = False
        self.pre_train = pre_train
        action_shape = len(self.actions)

        # set the device
        if use_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        self.agent = A3C(
            self.obs_shape,
            action_shape,
            device,
            critic_lr=1,
            actor_lr=1,
            random_rate=0,
        )

        if pre_train != None:
            config = torch.load(pre_train)
            self.agent.load_state_dict(config["best_agent"])

        self.agent.eval()
        self.reset()
        self.test_rewards = []
        self.sample_phase = 0

        # tell rover and referee it can go
        self.sleep(10)
        self.ready_pub.publish(True)

        self.agent_ready = True
        self.spin()

    def onState(self, states):
        total_num = 1000
        if not self.env_ready or not self.agent_ready:
            return 0

        if self.sample_phase == total_num:
            self.log(f"Test end with average reward: {np.array(self.test_rewards).mean()}")
            self.exit()
            return 0

        action = 0
        rewards = self.rewards
        self.test_rewards.append(rewards)
        self.log(f"[{self.sample_phase}/{total_num}]get reward {rewards}")
        if self.pre_train != None:
            (
                actions,
                action_log_probs,
                state_value_preds,
                entropy,
            ) = self.agent.select_action(states)
            action = actions.cpu().numpy()

        self.sample_phase += 1
        return action


if __name__ == "__main__":
    pre_train = "20230511122504.pt"
    Test(pre_train=pre_train, use_odom=True, use_KF=False, use_cuda=True)
