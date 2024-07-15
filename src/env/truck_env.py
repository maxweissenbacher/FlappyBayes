import gym
from gym import spaces
import numpy as np
from botorch.test_functions import Branin, Levy, Ackley, Hartmann, Michalewicz
import torch
from DrlPlatform.Models.abstract_server import AbstractServer
from DrlPlatform import UdpServer


import gym
import gym.envs
from DrlPlatform import UdpServer


def make_windtunnel_gym_env(flap_mode):
    """
    Generate a Gym environment which communicates with the wind tunnel
    experiment via UDP connection.

    Returns
    -------
    env : OpenAI gym environment

    """
    gym.envs.register(
        id='AhmedBody_AllObservations-v0',
        entry_point='Case.LABVIEW_Environment:AhmedBody_AllObservations'
    )

    env:gym.Env = gym.make(
        'AhmedBody_AllObservations-v0', 
        avg_window=12, 
        flap_mode=flap_mode,
    )
    #env = gym.make('AhmedBody_AllObservations-v0')
    env_server = UdpServer(
        server_host="192.168.1.192",
        server_port=16388,
        client_host="192.168.1.183",  #REPLACE WITH IP FROM PXI
        client_port=16387,
        package_timeout=5.0,
        max_package_size=16384
    )
    
    env_server.start_server()
    env.env.env_server = env_server  # type: ignore

    return env



if __name__ == '__main__':
    env = make_windtunnel_gym_env()
    state = env.reset()
    next_state, reward, done, info = env.step(torch.zeros(2))
