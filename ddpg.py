import actor_critic
import collections
import random
import math
import torch
import torch.nn
import numpy
import pathlib


def copy_weights(source: torch.nn.Module, target: torch.nn.Module, lr: float = 1):
    for src, dst in zip(source.parameters(), target.parameters()):
        dst.data.copy_(lr * src.data + (1 - lr) * dst.data)


class DeepDeterministicPolicyGradient:
    def __init__(self, n_states, n_actions, batch_size, hidden, noise_decay, discount_factor, update_lr, device='cpu'):
        self.ou_noise = OrnsteinUhlenbeckProcess(n_actions)
        self.replay_buffer = ReplayBuffer(1000000)
        self.training = True
        self.batch_size = batch_size
        self.noise_weight = 1
        self.noise_decay = 1 / noise_decay
        self.discount_factor = discount_factor
        self.update_lr = update_lr
        self.device = device
        self.actor = actor_critic.Actor(n_states, n_actions, hidden)
        self.actor_target = actor_critic.Actor(n_states, n_actions, hidden)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = actor_critic.Critic(n_states, n_actions, hidden)
        self.critic_target = actor_critic.Critic(n_states, n_actions, hidden)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.loss = torch.nn.MSELoss()
        copy_weights(self.actor, self.actor_target)
        copy_weights(self.critic, self.critic_target)
        self.to(device)

    def observe(self, state, action, next_state, reward, terminal):
        self.replay_buffer.push(state, action, next_state, reward, not terminal)

    def select_action(self, state, noise_decay=False):
        actor_training = self.actor.training
        if actor_training: self.actor.eval()
        with torch.no_grad():
            action = self.actor(torch.tensor(state, dtype=torch.float32, device=self.device)).cpu().numpy()
            if self.training:
                action += max(self.noise_weight, 0) * self.ou_noise()
            if noise_decay:
                self.noise_weight -= self.noise_decay
        if actor_training: self.actor.train()
        return numpy.clip(action, -1, 1)

    def update_parameters(self):
        batch = self.replay_buffer.sample(self.batch_size)
        state_batch = torch.tensor(numpy.stack(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(numpy.stack(batch.action), dtype=torch.float32, device=self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        terminal_batch = torch.tensor(batch.terminal, dtype=torch.bool, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(numpy.stack(batch.next_state), dtype=torch.float32, device=self.device)

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)
        expected_state_action_batch = reward_batch + (self.discount_factor * terminal_batch * next_state_action_values)

        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = self.loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        copy_weights(self.actor, self.actor_target, self.update_lr)
        copy_weights(self.critic, self.critic_target, self.update_lr)

        return value_loss.item(), policy_loss.item()

    def reset(self):
        self.ou_noise.reset()

    def train(self):
        self.training = True
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def save(self, path=''):
        torch.save(self.actor.state_dict(), pathlib.Path(path) / 'actor.torch')
        torch.save(self.critic.state_dict(), pathlib.Path(path) / 'critic.torch')

    def load(self, path=''):
        self.actor.load_state_dict(torch.load(pathlib.Path(path) / 'actor.torch'))
        self.critic.load_state_dict(torch.load(pathlib.Path(path) / 'critic.torch'))

    def to(self, device):
        self.device = device
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        return self


class ReplayBuffer:
    Transition = collections.namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = ReplayBuffer.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, min(batch_size, len(self)))
        return ReplayBuffer.Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)


class OrnsteinUhlenbeckProcess:
    # http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    def __init__(self, n_actions, sigma=0.3, theta=0.15, dt=1e-2, x0=None):
        self.mu = numpy.zeros(n_actions)
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.x_prev = None
        self.reset()

    def __call__(self):
        x = self.x_prev
        x += self.theta * (self.mu - self.x_prev) * self.dt
        x += self.sigma * numpy.sqrt(self.dt) * numpy.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else numpy.zeros_like(self.mu)
