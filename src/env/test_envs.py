import gym
from gym import spaces
import numpy as np
from botorch.test_functions import Branin, Levy, Ackley, Hartmann, Michalewicz
import torch


class TestEnv(gym.Env):
    metadata = {}

    def __init__(
            self,
            function='Branin',
            gaussian_noise_scale=0.0,
            ):

        super().__init__()
        if function == 'Branin':
            self.test_function = Branin(negate=True)
        elif function == 'Levy':
            self.test_function = Levy(dim=2, negate=True)
        elif function == 'Ackley':
            self.test_function = Ackley(dim=2, negate=True)
        elif function == 'Hartmann':
            self.test_function = Hartmann(dim=2, negate=True)
        elif function == 'Michalewicz':
            self.test_function = Michalewicz(dim=2, negate=True)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(2,))
        self.num_obs = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,))
        self.noise_scale = gaussian_noise_scale

    def step(self, action):
        observation = torch.zeros((self.num_obs,))
        reward = self.test_function(action) + self.noise_scale * torch.randn(())
        done = False
        return observation, reward, done, {}

    def reset(self):
        observation = torch.zeros((self.num_obs,))
        return observation


if __name__ == '__main__':
    env = TestEnv(function='Ackley', gaussian_noise_scale=2.0)
    env.step(torch.zeros(2))
