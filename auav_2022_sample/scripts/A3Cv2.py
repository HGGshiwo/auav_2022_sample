#!/usr/bin/env python3
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class A3C(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        random_rate: Propbility to choose the random action.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        random_rate: float,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()

        n_features = 4 * 6 * 8

        self.device = device
        self.n_actions = n_actions
        self.random_rate = random_rate
        self.state_que = torch.zeros((5, n_features))

        backbone_layers = [
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (240, 320, 8)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (120, 160, 16)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (60, 80, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (30, 40, 64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (15, 20, 64)
            nn.Conv2d(64, 8, kernel_size=(3, 2), stride=2, padding=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (4, 6, 8)
            nn.Flatten(0),  # (192)
        ]

        critic_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(
                32, n_actions
            ),  # estimate action logits (will be fed into a softmax later)
        ]

        critic_pre_layers = [
            PositionalEncoding(n_features),
            nn.LayerNorm(n_features),
        ]

        actor_pre_layers = [
            PositionalEncoding(n_features),
            nn.LayerNorm(n_features),
        ]

        # define actor and critic networks
        self.backbone = nn.Sequential(*backbone_layers).to(self.device)

        self.critic_pre = nn.Sequential(*critic_pre_layers).to(self.device)
        self.critic_atten = nn.MultiheadAttention(n_features, 4, 0.2).to(self.device)
        self.critic_fc = nn.Sequential(*critic_layers).to(self.device)

        self.actor_pre = nn.Sequential(*actor_pre_layers).to(self.device)
        self.actor_atten = nn.MultiheadAttention(n_features, 4, 0.2).to(self.device)
        self.actor_fc = nn.Sequential(*actor_layers).to(self.device)

        # define optimizers for actor and critic
        self.optim = optim.Adam(
            [
                {"params": self.backbone.parameters(), "lr": actor_lr * 1e-2},
                {"params": self.critic_pre.parameters(), "lr": critic_lr},
                {"params": self.critic_atten.parameters(), "lr": critic_lr},
                {"params": self.critic_fc.parameters(), "lr": critic_lr},
                {"params": self.actor_pre.parameters(), "lr": actor_lr},
                {"params": self.actor_atten.parameters(), "lr": actor_lr},
                {"params": self.actor_fc.parameters(), "lr": actor_lr},
            ]
        )

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values.
            action_logits_vec: A tensor with the action logits, with shape [n_actions].
        """
        x = torch.Tensor(x).to(self.device)
        feature = self.backbone(x)
        feature = feature.reshape((1, -1))

        critic_flow = self.critic_pre(feature)
        critic_flow, attn_output_weight = self.critic_atten(
            critic_flow, self.state_que, self.state_que
        )
        critic_flow = critic_flow + feature  # KF-like
        state_values = self.critic_fc(critic_flow)

        actor_flow = self.actor_pre(feature)
        actor_flow, attn_output_weight = self.actor_atten(
            actor_flow, self.state_que, self.state_que
        )
        actor_flow = actor_flow + feature  # KF-like
        action_logits_vec = self.actor_fc(actor_flow)[0]  # shape: (n_actions, )

        # put the feature into the state que
        self.state_que = torch.cat((self.state_que[1:], feature.detach()))

        return (state_values, action_logits_vec)

    def select_action(
        self, x: np.ndarray, use_random=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, ].
            state_values: A tensor with the state values, with shape [n_steps_per_update, ].
        """
        state_values, action_logits = self.forward(x)
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        if use_random:
            random_action_pd = torch.distributions.Categorical(
                torch.ones((self.n_actions)) / self.n_actions
            )
            random_action = random_action_pd.sample()
            choose_pd = torch.distributions.Categorical(
                torch.Tensor([self.random_rate, 1 - self.random_rate])
            )
            choose = choose_pd.sample()
            if choose == 0:
                actions = random_action
        entropy = action_pd.entropy()
        return (actions, action_log_probs, state_values, entropy)

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.optim.zero_grad()
        critic_loss.backward()
        self.optim.step()
