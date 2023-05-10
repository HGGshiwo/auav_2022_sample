#!/usr/bin/env python3
from Env3 import Env
from A2C import A2C
from A3Cv3 import A3C
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
from datetime import timedelta


class Train(Env):
    def __init__(self, pre_train) -> None:
        """
        Record the model data

        Args:
            pre_train:       pre_trained model path. Set to None if not use pretrain.
        """
        Env.__init__(self, use_odom=True)
        self.agent_ready = False
        self.pre_train = pre_train
        self.config = {
            "actor_lr": 1e-4,
            "critic_lr": 5e-3,
            "random_rate": 0.3,
            "n_updates": 10000,
            "n_steps_per_update": 2,
            "gamma": 0.999,
            "ent_coef": 0.01,  # coefficient for the entropy bonus (to encourage exploration)
            "lam": 0.95,  # hyperparameter for GAE
            "critic_losses": [],
            "actor_losses": [],
            "entropies": [],
            "return_queue": [],
            "use_cuda": True,
            "n_start": 0,
            "best_rewards": 0,
            "best_agent": {},
            "best_optim": {},
        }

        if self.pre_train != None:
            self.config = torch.load(pre_train)

        # agent hyperparams
        # Note: the actor has a slower learning rate so that the value targets become
        # more stationary and are theirfore easier to estimate for the critic
        self.actor_lr = self.config["actor_lr"]
        self.critic_lr = self.config["critic_lr"]
        self.random_rate = self.config["random_rate"]
        self.n_updates = self.config["n_updates"]
        self.n_steps_per_update = self.config["n_steps_per_update"]

        # environment hyperparams
        self.gamma = self.config["gamma"]
        self.ent_coef = self.config["ent_coef"]
        self.lam = self.config["lam"]  # hyperparameter for GAE
        self.critic_losses = self.config["critic_losses"]
        self.actor_losses = self.config["actor_losses"]
        self.entropies = self.config["entropies"]
        self.use_cuda = self.config["use_cuda"]
        self.sample_phase = self.config["n_start"]
        self.return_queue = self.config["return_queue"]
        self.step = 0
        # load pre-trained model if need.
        self.best_rewards = self.config["best_rewards"]
        self.best_agent = self.config["best_agent"]
        self.best_optim = self.config["best_optim"]

        self.start = time.perf_counter()
        obs_shape = self.obs_shape
        action_shape = len(self.actions)

        # set the device
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.ep_value_preds = torch.zeros(self.n_steps_per_update, device=self.device)
        self.ep_rewards = torch.zeros(self.n_steps_per_update, device=self.device)
        self.ep_action_log_probs = torch.zeros(
            self.n_steps_per_update, device=self.device
        )
        self.entropy = None
        self.masks = torch.zeros(self.n_steps_per_update)

        self.agent = A3C(
            obs_shape,
            action_shape,
            self.device,
            self.critic_lr,
            self.actor_lr,
            self.random_rate,
        )

        if pre_train != None:
            self.agent.load_state_dict(self.best_agent)
            self.agent.optim.load_state_dict(self.best_optim)

        self.log("start train")
        self.log(f"actor_lr: {self.actor_lr} critic_lr: {self.critic_lr}")
        self.log(f"use {self.device} for trainning")
        
        # tell rover and referee it can go
        self.sleep(10)
        self.ready_pub.publish(True)
        
        self.agent_ready = True
        self.spin()

    def save_data(self):
        if len(self.critic_losses) == 0:
            return

        now = int(round(time.time() * 1000))
        now_str = time.strftime("%Y%m%d%H%M%S", time.localtime(now / 1000))

        self.config["critic_losses"] = self.critic_losses
        self.config["actor_losses"] = self.actor_losses
        self.config["entropies"] = self.entropies
        self.config["best_rewards"] = self.best_rewards
        self.config["best_agent"] = self.best_agent
        self.config["best_optim"] = self.best_optim
        self.config["n_start"] = self.sample_phase
        self.config["return_queue"] = self.return_queue

        torch.save(self.config, f"{now_str}.pt")

        """plot the results"""
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))

        # episode return
        axs[0][0].set_title("Episode Returns")
        episode_returns = np.array(self.return_queue)
        axs[0][0].plot(np.arange(len(episode_returns)), episode_returns)
        axs[0][0].set_xlabel("Number of episodes")

        # entropy
        axs[1][0].set_title("Entropy")
        entropy_moving_average = np.array(self.entropies)
        axs[1][0].plot(entropy_moving_average)
        axs[1][0].set_xlabel("Number of updates")

        # critic loss
        axs[0][1].set_title("Critic Loss")
        critic_losses_moving_average = np.array(self.critic_losses)
        axs[0][1].plot(critic_losses_moving_average)
        axs[0][1].set_xlabel("Number of updates")

        # actor loss
        axs[1][1].set_title("Actor Loss")
        actor_losses_moving_average = np.array(self.actor_losses)
        axs[1][1].plot(actor_losses_moving_average)
        axs[1][1].set_xlabel("Number of updates")

        plt.tight_layout()

        plt.savefig(f"{now_str}.png")

    def onState(self, states):
        if not self.env_ready or not self.agent_ready:
            return 0

        # 查看训练是否结束
        if self.sample_phase == self.n_updates:
            end = time.perf_counter()
            self.log(f"train done in {timedelta(seconds=end-self.start)}")
            self.save_data()
            self.exit()
            return 0

        # 将当前的reward作为上一个action的reward
        last_step = self.n_steps_per_update - 1 if self.step == 0 else self.step - 1
        self.ep_rewards[last_step] = torch.tensor(self.rewards, device=self.device)
        self.log(f"step done with rewards: {self.rewards}")

        # 查看是否更新模型
        if self.step == 0 and self.entropy != None:
            critic_loss, actor_loss = self.agent.get_losses(
                self.ep_rewards,
                self.ep_action_log_probs,
                self.ep_value_preds,
                self.entropy,
                self.masks,
                self.gamma,
                self.lam,
                self.ent_coef,
                self.device,
            )

            # update the actor and critic networks
            self.agent.update_parameters(critic_loss, actor_loss)

            # log the losses and entropy
            self.critic_losses.append(critic_loss.detach().cpu().numpy())
            self.actor_losses.append(actor_loss.detach().cpu().numpy())
            self.entropies.append(self.entropy.detach().mean().cpu().numpy())

            self.log(
                f"[{self.sample_phase+1}/{self.n_updates}]actor loss: {self.actor_losses[-1]}, critic_loss: {self.critic_losses[-1]}"
            )

            if (
                len(self.return_queue) != 0
                and self.return_queue[-1] > self.best_rewards
            ):
                self.best_rewards = self.return_queue[-1]
                self.best_agent = copy.deepcopy(self.agent.state_dict())
                self.best_optim = copy.deepcopy(self.agent.optim.state_dict())

            # reset lists that collect experiences of an episode (sample phase)
            self.ep_value_preds = torch.zeros(
                self.n_steps_per_update, device=self.device
            )
            self.ep_rewards = torch.zeros(self.n_steps_per_update, device=self.device)
            self.ep_action_log_probs = torch.zeros(
                self.n_steps_per_update, device=self.device
            )
            self.masks = torch.zeros(self.n_steps_per_update)

            self.sample_phase += 1

        # 选择动作
        (
            actions,
            action_log_probs,
            state_value_preds,
            entropy,
        ) = self.agent.select_action(states)

        action = actions.cpu().numpy()

        self.ep_value_preds[self.step] = torch.squeeze(state_value_preds)
        self.ep_action_log_probs[self.step] = action_log_probs
        self.entropy = entropy

        # add a mask (for the return calculation later);
        # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
        self.masks[self.step] = torch.tensor(not self.terminated)

        self.step = (self.step + 1) % self.n_steps_per_update
        self.log(f"step with action {action}")
        return action


if __name__ == "__main__":
    pre_train = None

    # wait until train end.
    train = Train(pre_train)
