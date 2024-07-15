import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from pyDOE2 import lhs
from tqdm import tqdm
from src.plot.line_plot import line_plot_parameters
from src.plot.vertical_plot import vertical_plot
import pickle
import warnings
import logging


class GridSearch:
    def __init__(self, env, episode_steps, burnin_steps, reset_iterations, save_directory_path=None, console_output=False):
        self.N_DIMS = env.action_space.shape[0]
        self.env = env
        # TO-DO: figure out the lower and upper bounds for all the angles!
        # Consider that the flaps cannot touch and cannot overextend
        self.lower_bound = torch.tensor(env.action_space.low)
        self.upper_bound = torch.tensor(env.action_space.high)
        self.bounds_diff = self.upper_bound - self.lower_bound
        self.episode_time = episode_steps
        self.burnin_steps = burnin_steps
        self.total_steps = self.episode_time + self.burnin_steps
        self.console_output = console_output
        self.save_directory = save_directory_path
        self.reset_iterations = reset_iterations
        if save_directory_path is None:
            warnings.warn(f"save_directory_path parameter has not been specified. Results will not be saved.")
        
        logging.info(f'Using lower bounds {list(self.lower_bound.numpy())} and upper bounds {list(self.upper_bound.numpy())}.')

    def grid_search(self, angles): 
        train_y = []
        
        # Interactive plot of reward/performance
        plt.ion()
        fig, axs = plt.subplots()
        line2, = axs.plot([], [], color='black', marker='.')
        
        for i in range(len(angles)):
            if i % self.reset_iterations == 0:
                self.env.reset()
            x = angles[i]
            new_y, info = self.experiment(np.array([x]))
            train_y.append(new_y)
            
            if self.save_directory is not None:
                self.save(self.save_directory, torch.tensor(np.asarray(angles)).unsqueeze(-1), torch.tensor(np.asarray(train_y)).unsqueeze(-1), info, user_warnings=False)
            
            # Console logs
            console_output = f"Iteration {i}: "
            console_output += f"Angle {x:.3f}"
            console_output += ' | '
            console_output += f"Value {float(train_y[-1]):.5f}"
            logging.info(console_output)

            # Update plot
            line2.set_data(angles[:len(train_y)], train_y)
            axs.set_xlim(min(angles), max(angles))
            axs.set_ylim(1.2*min(train_y), 1.2*max(train_y))

            # Refresh the canvas
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        train_y = torch.tensor(np.asarray(train_y)).unsqueeze(-1)
        
        plt.ioff()
        plt.show(block=False)

        plt.plot(angles, train_y)
        plt.savefig(self.save_directory + 'angles_vs_reward.png', format='png', dpi=300)


    def experiment(self, angles):
        # Take repeated env steps and save all data received
        rewards = []
        observations = []

        for _ in range(self.total_steps):
            observation, reward, done, _ = self.env.step(action=angles)
            rewards.append(reward)
            observations.append(observation)

        # Discard the initial burn in steps
        croppped_rewards = rewards[self.burnin_steps:]

        # Extract mean (and an estimate of measurement noise?)
        mean_reward = torch.tensor(np.mean(np.asarray(croppped_rewards)))

        # Store any desired extra information in an info dictionary
        info = {'rewards': rewards, 'observations': observations}

        return mean_reward, info

    @staticmethod
    def save(path, train_x, train_y, info, user_warnings=True):
        import os
        if not (os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
            if user_warnings:
                warnings.warn(f"Directory {path} did not exist. Created this directory.")
        
        # Save angles and episodic rewards
        filepath = os.path.join(path, "BO_output_train_x.dat")
        if user_warnings:
            if os.path.exists(filepath) and os.path.isfile(filepath):
                warnings.warn(f"{filepath} exists and will be overwritten.")    
        with open(filepath, 'wb') as f:
            np.savetxt(f, train_x.numpy())
        
        # Save angles and episodic rewards
        filepath = os.path.join(path, "BO_output_train_y.dat")
        if user_warnings:
            if os.path.exists(filepath) and os.path.isfile(filepath):
                warnings.warn(f"{filepath} exists and will be overwritten.")    
        with open(filepath, 'wb') as f:
            np.savetxt(f, train_y.numpy())
        
        
        # Save pressure observations
        filepath = os.path.join(path, "pressure_logs.dat")
        if user_warnings:
            if os.path.exists(filepath) and os.path.isfile(filepath):
                warnings.warn(f"{filepath} exists and will be overwritten.")
        with open(filepath, 'ab') as f:
            np.savetxt(f, info['observations'])
            
        # Save rewards
        filepath = os.path.join(path, "reward_logs.dat")
        if user_warnings:
            if os.path.exists(filepath) and os.path.isfile(filepath):
                warnings.warn(f"{filepath} exists and will be overwritten.")
        with open(filepath, 'ab') as f:
            np.savetxt(f, info['rewards'])
        
        return None

