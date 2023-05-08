#!/usr/bin/env python3
from Env import Env
from A2C import A2C
import torch
import time
from A3C import A3C
import matplotlib.pyplot as plt
import numpy as np
import copy
from datetime import timedelta


class State:
    def __init__(self, pre_train) -> None:
        """
        Record the model data

        Args:
            pre_train:       pre_trained model path. Set to None if not use pretrain.
        """
        self.pre_train = pre_train
        self.config = {
            "actor_lr": 1e-5,
            "critic_lr": 5e-5,
            "random_rate": 0.3,
            "n_updates": 800,
            "n_steps_per_update": 4,
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

    def __enter__(self):
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
        self.n_start = self.config["n_start"]
        self.return_queue = self.config["return_queue"]

        # load pre-trained model if need.
        self.best_rewards = self.config["best_rewards"]
        self.best_agent = self.config["best_agent"]
        self.best_optim = self.config["best_optim"]

        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if exception_type != None:
            print(exception_traceback)

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
        self.config["n_start"] = self.n_start
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


if __name__ == "__main__":
    pre_train = None

    with State(pre_train) as state:
        start = time.perf_counter()
        env = Env(use_odom=True)
        obs_shape = env.obs.shape
        action_shape = len(env.actions)

        # set the device
        if state.use_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # init the agent
        agent = A3C(
            obs_shape,
            action_shape,
            device,
            state.critic_lr,
            state.actor_lr,
            state.random_rate,
        )
        if pre_train != None:
            agent.load_state_dict(state.best_agent)
            agent.optim.load_state_dict(state.best_optim)

        env.log("start train")
        env.log(f"actor_lr: {state.actor_lr}")
        env.log(f"critic_lr: {state.critic_lr}")
        env.log(f"use {device} for trainning")

        for sample_phase in range(state.n_start, state.n_updates):
            # we don't have to reset the envs, they just continue playing
            # until the episode is over and then reset automatically

            # reset lists that collect experiences of an episode (sample phase)
            ep_value_preds = torch.zeros(state.n_steps_per_update, device=device)
            ep_rewards = torch.zeros(state.n_steps_per_update, device=device)
            ep_action_log_probs = torch.zeros(state.n_steps_per_update, device=device)
            masks = torch.zeros(state.n_steps_per_update)

            # at the start of training reset all envs to get an initial state
            if sample_phase == 0:
                states, info = env.reset()

            # play n steps in our parallel environments to collect data
            for step in range(state.n_steps_per_update):
                # select an action A_{t} using S_{t} as input for the agent
                (
                    actions,
                    action_log_probs,
                    state_value_preds,
                    entropy,
                ) = agent.select_action(states)

                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                states, rewards, terminated, truncated, infos = env.step(
                    actions.cpu().numpy()
                )

                ep_value_preds[step] = torch.squeeze(state_value_preds)
                ep_rewards[step] = torch.tensor(rewards, device=device)
                ep_action_log_probs[step] = action_log_probs

                # add a mask (for the return calculation later);
                # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                masks[step] = torch.tensor(not terminated)

            # calculate the losses for actor and critic
            critic_loss, actor_loss = agent.get_losses(
                ep_rewards,
                ep_action_log_probs,
                ep_value_preds,
                entropy,
                masks,
                state.gamma,
                state.lam,
                state.ent_coef,
                device,
            )

            # update the actor and critic networks
            agent.update_parameters(critic_loss, actor_loss)

            # log the losses and entropy
            state.critic_losses.append(critic_loss.detach().cpu().numpy())
            state.actor_losses.append(actor_loss.detach().cpu().numpy())
            state.entropies.append(entropy.detach().mean().cpu().numpy())

            env.log(
                f"[{sample_phase+1}/{state.n_updates}]actor loss: {state.actor_losses[-1]}, critic_loss: {state.critic_losses[-1]}"
            )

            if len(env.return_queue) != 0 and env.return_queue[-1] > state.best_rewards:
                state.best_rewards = env.return_queue[-1]
                state.best_agent = copy.deepcopy(agent.state_dict())
                state.best_optim = copy.deepcopy(agent.optim.state_dict())
                state.n_start = sample_phase
        state.return_queue = env.return_queue

        end = time.perf_counter()
        env.log(f"train done in {timedelta(seconds=end-start)}")
