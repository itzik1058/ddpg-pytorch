import torch
import torch.nn as nn


class StateEmbedding(nn.Module):
    def __init__(self, n_states, hidden):
        super(StateEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True)
        )

    def forward(self, state):
        return self.embed(state)


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden):
        super(Actor, self).__init__()
        self.state_embed = StateEmbedding(n_states, hidden)
        self.action = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
            nn.Tanh()
        )

    def forward(self, state):
        state_embed = self.state_embed(state)
        return self.action(state_embed)


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden):
        super(Critic, self).__init__()
        self.state_embed = StateEmbedding(n_states, hidden)
        self.reward = nn.Sequential(
            nn.Linear(hidden + n_actions, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, action):
        state_embed = self.state_embed(state)
        return self.reward(torch.cat([state_embed, action], dim=1))
