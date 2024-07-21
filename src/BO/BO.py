import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
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
from botorch.test_functions import Ackley


class BOTrainer:
    def __init__(
            self,
            n_initial_samples,
            n_dims,
            lower_bounds,
            upper_bounds,
            beta,
            save_directory_path=None,
            console_output=False,
            debug=True):
        self.N_DIMS = n_dims
        self.lower_bound = lower_bounds
        self.upper_bound = upper_bounds
        if not self.lower_bound.shape == self.upper_bound.shape:
            raise ValueError(f"Lower and upper bound must have the same shape.")
        if not self.lower_bound.shape[0] == self.N_DIMS:
            raise ValueError(f"Lower and upper bound must be arrays of shape [{self.N_DIMS}]. Got shape {self.lower_bound.shape}.")
        self.bounds_diff = self.upper_bound - self.lower_bound
        self.beta = beta
        self.n_initial_samples = n_initial_samples
        self.console_output = console_output
        self.save_directory = save_directory_path
        self.debug = debug
        if save_directory_path is None:
            warnings.warn(f"save_directory_path parameter has not been specified. Results will not be saved.")
        
        logging.info(f'Using lower bounds {list(self.lower_bound.numpy())} and upper bounds {list(self.upper_bound.numpy())}.')

    def initialize_observations(self, save=True):
        # Initialize observations
        lhs_samples = lhs(self.N_DIMS, samples=self.n_initial_samples)
        train_x = torch.tensor(lhs_samples, dtype=torch.double).mul(self.bounds_diff).add(self.lower_bound)
        train_y = []
        for i in range(train_x.shape[0]):
            params = train_x[i]
            new_y, info = self.experiment(params)
            train_y.append(new_y)
            if save and self.save_directory is not None:
                self.save(self.save_directory, train_x, train_y, info, user_warnings=False)

            # Console logs
            console_output = f"Iteration {i}: "
            console_output += "params "
            console_output += ' / '.join([f"{v:.3f}" for v in params.numpy()])
            console_output += ' | '
            console_output += f"Value {float(train_y[-1]):.5f}"
            logging.info(console_output)

        train_y = np.asarray(train_y, dtype=np.float64)
        # Convert the NumPy array to a PyTorch tensor and add a new dimension
        train_y = torch.tensor(train_y).unsqueeze(-1)

        return train_x, train_y

    def train_one_step(self, train_x, train_y):
        scaled_y = (train_y - train_y.mean())/(train_y.std())
        scaled_x = (train_x-self.lower_bound)/self.bounds_diff

        # The model seems to converge better when we don't scale train_y...
        model = SingleTaskGP(scaled_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Define and optimise acquisition function
        UCB = UpperConfidenceBound(model, beta=self.beta)
        unit_bounds = torch.stack([torch.zeros(self.N_DIMS), torch.ones(self.N_DIMS)])
        new_x_scaled, _ = optimize_acqf(
            acq_function=UCB,
            bounds=unit_bounds,
            q=1,
            num_restarts=10,
            raw_samples=self.n_initial_samples,
        )
        # Run experiment
        new_x = self.bounds_diff * new_x_scaled + self.lower_bound
        # print(f"Try the following parameters: "{new_x})
        new_y, info = self.experiment(new_x.flatten())

        new_y = torch.tensor(new_y)
        new_y = new_y.view([1, 1])
        # Add to training data
        new_train_x = torch.cat([train_x, new_x])
        new_train_y = torch.cat([train_y, new_y])

        return new_train_x, new_train_y, info

    def train(self, init_train_x, init_train_y, OPTIMISATION_ITERATIONS, save=True):
        train_x = init_train_x
        train_y = init_train_y

        # Interactive plot of reward/performance
        plt.ion()
        fig, axs = plt.subplots()
        line2, = axs.plot([], [], color='black', marker='.')

        # Bayesian Optimization loop
        for i in range(OPTIMISATION_ITERATIONS):
            train_x, train_y, info = self.train_one_step(train_x, train_y)
            if save and self.save_directory is not None:
                self.save(self.save_directory, train_x, train_y, info, user_warnings=False)

            # Console logs
            console_output = f"Iteration {i}: "
            console_output += "params "
            console_output += ' / '.join([f"{v:.3f}" for v in train_x[-1].numpy()])
            console_output += ' | '
            console_output += f"Value {float(train_y[-1]):.5f}"
            logging.info(console_output)

            # Update plot
            line2.set_data(np.arange(len(train_y)), train_y)
            axs.set_xlim(0, len(train_y) - 1)
            axs.set_ylim(1.2*min(train_y), 1.2*max(train_y))
            axs.axvspan(xmin=0, xmax=init_train_x.shape[0] - 1.0, color='green')

            # Refresh the canvas
            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.ioff()
        plt.show(block=False)

        vertical_plot(
            train_x,
            train_y,
            init_train_x.shape[0],
            save_directory=self.save_directory
        )

        return train_x, train_y

    def experiment(self, params):
        # Here, all the interaction with the fluids solver occurs.
        # You need to set off a run with the correct parameters
        # Then wait for the run to finish and write its results into a file
        # Then read that file here and compute the drag coefficient
        import os
    
        if self.debug:
            test_function = Ackley(dim=self.N_DIMS, noise_std=0.1, negate=True)
            drag = test_function(params)
        else:
            # print(f"Try the following parameters: ",{params})
            with open('Parameters.txt', 'w') as file:
                # Join tensor values into a single line with spaces separating them
                file.write(' '.join(map(str, params.tolist())))
            # Once the actual simulation code is here, remove this error message!
            Windows=True
            if Windows:
                subprocess.run(["prueba.bat"],shell=True)
            else:
                subprocess.run(["./starccm", "-batch", "params.java", os.path.join(".","With_Flaps_RANS_Fine_1.sim")], shell=True) 
            # Path to your CSV file
            file_path = os.path.join(".","Cd.csv")

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Get the value from the second column and last row as a string
            drag = str(df.iloc[-1, 1])

            try:
                drag = float(drag)  # Convert input to a float (handles integers as well)
            except ValueError:
                print(f"Invalid input. Please enter a valid number.")
        
        # raise ValueError(f"Simulation mode not implemented. Debug mode must be activated.")

        # Store any desired extra information in an info dictionary
        # Important: every entry is expected to be a NUMPY ARRAY
        info = {}

        return drag, info

    @staticmethod
    def save(path, train_x, train_y, info, user_warnings=True):
        import os
        if not (os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
            if user_warnings:
                warnings.warn(f"Directory {path} did not exist. Created this directory.")
        
        # Save params and episodic rewards
        filepath = os.path.join(path, "BO_output.pkl")
        if user_warnings:
            if os.path.exists(filepath) and os.path.isfile(filepath):
                warnings.warn(f"{filepath} exists and will be overwritten.")
        output_dict = {'train_x': train_x, 'train_y': train_y, 'info': info}

        with open(filepath, 'wb') as f:
            pickle.dump(output_dict, f)
        
        # Save info - each entry is saved separately
        for key, arr in info.items():
            filepath = os.path.join(path, f"{key}.dat")
            if user_warnings:
                if os.path.exists(filepath) and os.path.isfile(filepath):
                    warnings.warn(f"{filepath} exists and will be overwritten.")
            with open(filepath, 'ab') as f:
                np.savetxt(f, arr)
        
        return None

