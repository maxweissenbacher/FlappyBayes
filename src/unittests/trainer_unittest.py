from src.BO.BO import BOTrainer
from src.env.test_envs import TestEnv
import warnings
from botorch.exceptions import InputDataWarning


def unittest_trainer_test_env(fn_name):
    # Ignore warnings about input scaling
    warnings.filterwarnings('ignore', category=InputDataWarning)

    env = TestEnv(function=fn_name)
    init_num_obs = 20
    trainer = BOTrainer(env, n_initial_samples=init_num_obs, save_directory_path='./test_outputs/')

    # Check parameter bounds
    if not (trainer.lower_bound.numpy() == env.action_space.low).all():
        raise RuntimeError("Lower bounds do not match. Check parameter bounds.")
    if not (trainer.upper_bound.numpy() == env.action_space.high).all():
        raise RuntimeError("Upper bounds do not match. Check parameter bounds.")

    # Initialise parameters
    init_train_x, init_train_y, info = trainer.initialize_observations()

    if not (init_train_y.shape[0] == init_num_obs and init_train_y.shape[1] == 1):
        raise RuntimeError("Got wrong shape for init rewards.")

    train_steps = 25
    train_x, train_y, info = trainer.train(init_train_x, init_train_y, info, train_steps)

    if not train_x.shape[0] == init_train_y.shape[0]+train_steps:
        raise RuntimeError("Final number of data points is wrong.")


if __name__ == '__main__':
    unittest_trainer_test_env('Branin')
    unittest_trainer_test_env('Levy')
