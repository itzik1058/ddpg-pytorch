import environment
import ddpg
import gym
import torch
import numpy
import time
import matplotlib.pyplot as plt


def train(n_episode=1000, warmup=100, batch_size=64, hidden=128, discount_factor=0.99, update_lr=0.001,
          max_episode_steps=500, device='cpu', render=False, save=False):
    env = environment.NormalizedActionEnvironment(gym.make('Pendulum-v0'))
    env.seed(0)
    torch.manual_seed(0)
    numpy.random.seed(0)
    agent = ddpg.DeepDeterministicPolicyGradient(
        env.observation_space.shape[0], env.action_space.shape[0], batch_size=batch_size,
        hidden=hidden, noise_decay=5000, discount_factor=discount_factor, update_lr=update_lr
    )
    agent.to(device)
    rewards = []
    total_steps = 0
    for episode in range(n_episode):
        start = time.time()
        episode_rewards = []
        episode_steps = 0
        agent.reset()
        state = env.reset()
        agent.train()
        done = False
        while not done:
            if total_steps < warmup:
                action = numpy.random.uniform(-1, 1, env.action_space.shape)
            else:
                action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if render: env.render()
            agent.observe(state, action, next_state, reward, done)
            if total_steps >= warmup:
                agent.update_parameters()
            state = next_state

            episode_rewards.append(reward.item())
            episode_steps += 1
            total_steps += 1
            if max_episode_steps and episode_steps >= max_episode_steps:
                done = True

        # print(f'Episode {episode} Reward {episode_reward}')
        rewards.append(episode_rewards)
        if episode % 10 == 0:
            episode_rewards = []
            agent.reset()
            state = env.reset()
            agent.eval()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                if render: env.render()
                state = next_state
                episode_rewards.append(reward.item())
            # print(f'Episode {episode} Test Reward {episode_reward}')
            rewards.append(episode_rewards)
        if save: agent.save()
        rewards_arr = numpy.array(rewards)
        rewards_sum = numpy.sum(rewards_arr, axis=1)
        print(f'Episode {episode} Total steps {total_steps} Time {time.time() - start:.2f}s '
              f'Reward {rewards_sum[-1]} Average Reward {numpy.mean(rewards_sum[-10:])}')
        plt.plot(rewards_sum)
        plt.xlabel('Episode')
        plt.title('Reward')
        plt.savefig('rewards.png')
        plt.close()


if __name__ == '__main__':
    train(device='cuda:0', render=True)
