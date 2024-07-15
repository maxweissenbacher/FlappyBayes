from src.BO.BO import BOTrainer
from src.env.test_envs import TestEnv
import warnings
from botorch.exceptions import InputDataWarning
from src.plot.line_plot import line_plot_parameters
from src.plot.vertical_plot import vertical_plot


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

    line_plot_parameters(train_x, train_y, N_INITIAL_SAMPLES=init_num_obs)

    vertical_plot(train_x, train_y, N_INITIAL_SAMPLES=init_num_obs)

