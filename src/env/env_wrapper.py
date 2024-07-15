import torch
from test_envs import TestEnv
import numpy as np


def experiment(env, angles, num_env_steps=1000):
    # Take repeated env steps and save all data received
    rewards = []
    observations = []
    for _ in range(num_env_steps):
        observation, reward, done, _ = env.step(action=angles)
        rewards.append(reward.cpu().numpy())
        observations.append(observation.cpu().numpy())

    # Extract mean (and an estimate of measurement noise?)
    mean_reward = np.mean(np.asarray(rewards))

    return mean_reward


if __name__ == '__main__':
    env = TestEnv(function='Ackley', gaussian_noise_scale=1.0)
    experiment(env, torch.zeros(2))

