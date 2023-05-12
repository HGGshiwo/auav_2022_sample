#!/usr/bin/env python3
from Env5 import Env
from A2C import A2C
from A3Cv3 import A3C as A3Cv3
import torch
import numpy as np


class Test(Env):
    def __init__(
        self,
        pre_train,
        use_cuda=True,
        use_odom=True,
        use_KF=False,
        use_RL=True,
        state_mode="img",
        action_mode="vel",
    ):
        super().__init__(
            use_odom=use_odom,
            use_KF=use_KF,
            use_RL=use_RL,
            verbose=True,
            state_mode=state_mode,
            action_mode=action_mode,
        )
        self.agent_ready = False
        self.pre_train = pre_train

        # set the device
        if use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.agent = None
        if self.state_mode == "img":
            self.agent = A2C(
                self.obs_shape,
                self.action_shape,
                self.device,
                critic_lr=1,
                actor_lr=1,
                random_rate=0,
            )
        else:
            self.agent = A3Cv3(
                self.obs_shape,
                self.action_shape,
                self.device,
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
        if not self.env_ready or not self.agent_ready:
            return 0

        if self.episode_num > 1:
            self.log(
                f"Test end with average reward: {np.array(self.test_rewards).mean()}"
            )
            self.exit()
            return 0

        action = 0
        rewards = self.rewards
        self.test_rewards.append(rewards)
        self.log(f"get reward {rewards}")
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
    pre_train = None
    state_mode = "pos"
    action_mode = "vel"
    Test(
        pre_train=pre_train,
        use_odom=True,
        use_KF=False,
        use_cuda=True,
        use_RL=False,
        state_mode=state_mode,
        action_mode=action_mode,
    )
