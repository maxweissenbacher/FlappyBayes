import matplotlib.pyplot as plt
import numpy as np
from src.BO.BO import BOTrainer
from src.env.test_envs import TestEnv
import warnings
from botorch.exceptions import InputDataWarning


def scatter_plot_BO_points(train_x, train_y, N_INITIAL_SAMPLES):
    fig, axs = plt.subplots(2, 1,
                            figsize=(5, 8),
                            constrained_layout=True)
    """
    # Generate analytical response surface
    # This is great for testing but not possible in experiment
    # TO-DO: add an ESTIMATED analytical response surface?
    x = np.linspace(bounds[0, 0], bounds[1, 0], 100)
    y = np.linspace(bounds[0, 1], bounds[1, 1], 100)
    X, Y = np.meshgrid(x, y)
    Z = experiment(torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=dtype), function=FUNCTION).numpy().reshape(100, 100)
    cp = axs[0].contourf(X, Y, Z, levels=50, cmap='inferno')
    fig.colorbar(cp)
    """

    # Plot experiment locations
    axs[0].scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(),
                   marker='x', c='g', s=50, label='BO Points')
    axs[0].scatter(train_x[:N_INITIAL_SAMPLES, 0].numpy(), train_x[:N_INITIAL_SAMPLES, 1].numpy(),
                   marker='x', c='b', s=50, label='Initial Points')
    axs[0].set_title('Function with Sampled Points')
    axs[0].set_xlabel(r'$x_1$')
    axs[0].set_ylabel(r'$x_2$')
    axs[0].legend(loc='upper left')

    # Plot iterations
    axs[1].plot(train_y[N_INITIAL_SAMPLES:])
    axs[1].set_xlabel('iteration')
    axs[1].set_ylabel(r'$y$')

    plt.show()


if __name__ == '__main__':
    # Ignore warnings about input scaling
    warnings.filterwarnings('ignore', category=InputDataWarning)

    env = TestEnv(function='Branin')
    init_num_obs = 10
    trainer = BOTrainer(env, n_initial_samples=init_num_obs, save_directory_path=None)

    # Initialise parameters
    init_train_x, init_train_y, info = trainer.initialize_observations()

    train_steps = 20
    train_x, train_y, info = trainer.train(init_train_x, init_train_y, info, train_steps)

    scatter_plot_BO_points(train_x, train_y, N_INITIAL_SAMPLES=init_num_obs)

