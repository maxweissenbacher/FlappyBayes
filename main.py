from src.BO.BO import BOTrainer
import warnings
from botorch.exceptions import InputDataWarning
import hydra
import logging
import torch

import os
import sys
sys.path.append(os.getcwd() + '/src/env/')


@hydra.main(config_path="./", config_name="config", version_base="1.2")
def main(cfg: "DictConfig"):
    # Ignore warnings about input scaling
    warnings.filterwarnings('ignore', category=InputDataWarning)

    # Define BO trainer
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'
    trainer = BOTrainer(
        n_initial_samples=cfg.optim.init_num_obs,
        n_dims=cfg.optim.n_parameters,
        lower_bounds=torch.tensor(cfg.optim.lower_bounds),
        upper_bounds=torch.tensor(cfg.optim.upper_bounds),
        beta=cfg.optim.beta,
        save_directory_path=output_dir,
        console_output=cfg.logging.console_output,
        debug=cfg.optim.debug,
    )
    if trainer.save_directory:
        logging.info(f"Logging files to {output_dir}.")

    # Initialise parameters
    logging.info("--------------------")
    logging.info(f"Initial random parameter search...")
    logging.info("--------------------")
    init_train_x, init_train_y = trainer.initialize_observations()

    logging.info("--------------------")
    logging.info(f"Optimising angles...")
    logging.info("--------------------")
    train_x, train_y = trainer.train(init_train_x, init_train_y, cfg.optim.train_steps)

    logging.info(f"Finished execution.")


if __name__ == '__main__':
    main()
