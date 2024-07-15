import os
import logging
import numpy as np

from gym import spaces
from DrlPlatform import PlatformEnvironment
import time
from scipy.signal import welch, detrend, savgol_filter
import matplotlib.pyplot as plt


#from pykalman import KalmanFilter
     
class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')  # Initialise ring array 'data' as length-array of floats
        self.index = 0  # Initialise InPointer as 0 (where new data begins to be written)

    def extend(self, x):
        "adds array x to ring buffer"
        x_indices = (self.index + np.arange(x.size)) % self.data.size  # Find indices that x will occupy in 'data' array
        self.data[x_indices] = x  # Input the new array into ring buffer ('data')
        self.index = x_indices[-1] + 1  # Find new index for next new data

    def get(self):
        "Returns the first-in-first-out data in the ring buffer (returns data in order of introduction)"
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]

    def get_value(self, idx):
        "Returns the value at the given index in FIFO order"
        # Compute the actual index in the underlying data array
        actual_idx = (self.index + idx) % self.data.size
        #print('data size is')
        #print(self.data.size)
        return self.data[actual_idx]
    
start = time.time()

class AhmedBody_AllObservations(PlatformEnvironment):

    def __init__(self, avg_window, flap_mode):

        super(AhmedBody_AllObservations, self).__init__()
        
        self.size_history = 20000 # Size of the RingBuffer

        self.total_steps = int(1500)

        self.obs_with_force = False # Whether include drag force in observations
        self.include_anlge = True # Whether include angle in observations
        self.include_action = True
        self.pdiff_obs = False
        self.append_pdiff = True
        self.total_drag = 0
        self.total_side_force = 0
        self.total_lift = 0



        self.baseline_bp = 0
        self.baseline_pdiff = 0
        self.encoder1=0
        self.episodic_bp = 0
        self.episodic_pdiff = 0
        self.target_frequency = 0
        self.total_actionPF = 0
        self.periodic_window = 200
        self.steps_action_augmented = 5
        
       
        
        self.flap_behavior = flap_mode # Whether use snaking, clapping, free, LR_symmetric
        self.interval_record_baseline = 1 # Steps interval to extend reset time and obtain forces in baseline, which measures load cell drifting
        #self.baseline_duras = 500  # Calculate baseline for 0.02 * baseline_duras seconds
        self.reward_function = 'bo_reward_avg' #can also use 'force_reward' 'base_pressure_reward'
        self.twoflapmodes = 'LR'
        self.avg_window_max = avg_window
        
        
        # Create ring buffers to hold recent history of jet values, probe values, lift, drag, area
        self.history_parameters = {}
        
        # Variables store forces drifting from baseline flow, and compensation for reward functions
        self.avg_baseline_drag = np.array([])
        self.avg_baseline_lift = np.array([])
        self.avg_baseline_side = np.array([])
        self.avg_baseline_pressure = np.array([])
        self.avg_baseline_pdiff = np.array([])
        
        self.reward_compensation = 0

    
            

        self.reward_shaping_trigger = False

        
        
        # Buffer for power and actions (voltage)
        for num_flaps in range(4):
            self.history_parameters["voltage_{}".format(num_flaps)] = RingBuffer(self.size_history)
            self.history_parameters["power_{}".format(num_flaps)] = RingBuffer(self.size_history)
            self.history_parameters["action_{}".format(num_flaps)] = RingBuffer(self.size_history)
            if self.include_anlge == True:
                self.history_parameters["angles_{}".format(num_flaps)] = RingBuffer(self.size_history)
                
                     
        # Buffer for forces
        self.history_parameters["drag"] = RingBuffer(self.size_history)
        self.history_parameters["lift"] = RingBuffer(self.size_history)
        self.history_parameters["side"] = RingBuffer(self.size_history)
        self.history_parameters["base pressure"] = RingBuffer(self.size_history)
        self.history_parameters["pdiff_abs"] = RingBuffer(self.size_history)
        self.history_parameters["encoder1"] = RingBuffer(self.size_history)
        self.history_parameters["action_top"] = RingBuffer(self.size_history)
        self.history_parameters["action_bottom"] = RingBuffer(self.size_history)
        self.history_parameters["action_left"] = RingBuffer(self.size_history)
        
        
        self._logger = logging.getLogger(__name__)
        self.pressure_max = np.finfo(np.float64).max
        self.ForceX_max = np.finfo(np.float64).max
        self.ForceY_max = np.finfo(np.float64).max
        self.ForceZ_max = np.finfo(np.float64).max
        self.power_max = np.finfo(np.float64).max
        self.voltage_max = np.finfo(np.float64).max
        
        if self.flap_behavior == 'LR_only_symmetric':
            self.action_max = np.asarray([10.])  # left/right
        elif self.flap_behavior == 'test':
            self.action_max = np.asarray([10., 5., 8., 8.])  # top, bottom, right, left
        elif self.flap_behavior == 'LR_symmetric':
            self.action_max = np.asarray([10., 8., 8.])  # top, bottom, right/left
        elif self.flap_behavior == 'bottom_only':
            self.action_max = np.asarray([9.])  # bottom
        elif self.flap_behavior == 'top_only':
            self.action_max = np.asarray([10.])  # top
        elif self.flap_behavior == 'TB_only':
            self.action_max = np.asarray([10., 9.])  # top, bottom
        else:
            raise RuntimeError(f'Flap mode not defined correctly. Got {self.flap_behavior}')
        
        if self.pdiff_obs == True:
            self.state_shape = 28
        else:
            self.state_shape = 55
        
         
            

        
        if self.obs_with_force == True:
            self.state_shape += 2
            
            
        

        if self.flap_behavior == 'snaking' or self.flap_behavior == 'clapping':
            self.action_shape = 2
            
        elif self.flap_behavior == 'iso_snaking' or self.flap_behavior == 'iso_clapping':
            self.action_shape = 3
            
        elif self.flap_behavior == 'free':
            self.action_shape = 4

        elif self.flap_behavior == 'test':
            self.action_shape = 4
        
        elif self.flap_behavior == 'LR_symmetric':
            self.action_shape = 3
            
        elif self.flap_behavior == 'LR_only_symmetric':
            self.action_shape = 1
            
        elif self.flap_behavior == 'TB_only':
            self.action_shape = 2
            
        elif self.flap_behavior == 'top_only':
            self.action_shape = 1
            
        elif self.flap_behavior == 'bottom_only':
            self.action_shape = 1
            
        else:
            assert 'Flap behavior is not defiend correctly'
            
       

        #n_dim_state = 11 # 7 pressures + 4 actions
        #n_dim_obs = 11

        if self.include_action == True:
            self.state_shape +=  self.action_shape * self.steps_action_augmented
            
        if self.include_anlge == True:
            self.state_shape +=  self.action_shape

        limit_obs = np.array([])
        
        for nums in range(self.state_shape):
            limit_obs = np.append(limit_obs, self.pressure_max)
            
        limit_act = np.array([])
        for nums in range(self.action_shape):
            limit_act = np.append(limit_act, self.action_max[nums])

        self.observation_space = spaces.Box(-limit_obs, limit_obs, dtype=np.float64)  # type: ignore
        self.action_space = spaces.Box(-limit_act, limit_act, dtype=np.float64)
        self.total_reward = 0.0


        self.done = False
        self.n_episodes = -1
        self.prev_time = 0
        
        # Historical data
        self.previous_pressure = {}
        self.previous_ForceX = None
        self.previous_ForceY= None
        self.previous_ForceZ = None
        self.current_ForceX = None
        self.current_ForceY = None
        self.current_ForceZ = None
        self.previous_power = {}
        self.previous_voltage = {}
        self.previous_action = {}
        self.previous_angles = {}
        self.mean_basepressure = None
        self.previous_abspdiff = None
        self.all_indices = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                            31,32,33,34,35,36,37,38,39,40,41,45,46,47,48,49,50,54,55,56,57,58,59,63,64,65]

        
    def write_history_parameters(self):
        '''
        Add data of last step to history buffers
        :return:
        '''

        self.history_parameters["drag"].extend(np.array(self.current_ForceX))
        self.history_parameters["side"].extend(np.array(self.current_ForceY))
        self.history_parameters["lift"].extend(np.array(self.current_ForceZ))

        
        self.history_parameters["mean_basepressure"].extend(np.array(self.mean_basepressure))
        self.history_parameters["pdiff_abs"].extend(np.array(self.meanabs_pdiff))
        self.history_parameters["encoder1"] .extend(np.array(self.encoder1))
        

        self.history_parameters["action_top"].extend(np.array([self.previous_action[0]]))
        self.history_parameters["action_bottom"].extend(np.array([self.previous_action[1]]))
        self.history_parameters["action_right"].extend(np.array([self.previous_action[2]]))
        self.history_parameters["action_left"].extend(np.array([self.previous_action[3]]))
        
        
        
        
    def _receive_payload(self):

        if self.include_anlge == True:
            #payload = self.env_server.receive_payload(">dddddddddddddddddddddd")
            num_d = 71  # Specify the number of 'd' characters you need
            format_string = ">" + "d" * num_d
            payload = self.env_server.receive_payload(format_string)


        formatted_payload = self.format_payload(payload)
        #print(formatted_payload)
        
 

        return formatted_payload
    

    
    def step(self, action):
        # Negative reward for a step
        info = {}
        info['indicator'] = False
        #print(action)
        #print(action2)
        self.reward_shaping_trigger = False
        #Ahmed_section = 0.216 * 0.160
        Velocity = 15
        self.n_step += 1
        # Sending action through env server3

        if action is None:
            no_flaps=4
            actions = np.zeros((no_flaps,))
            
        if self.flap_behavior == 'snaking':
            actions = np.array([action[0],-action[0],action[1],-action[1]])
            
        elif self.flap_behavior == 'clapping':
            actions = np.array([action[0],action[0],action[1],action[1]])
            
        elif self.flap_behavior == 'iso_snaking':
            actions = np.array([action[0],-action[0],action[1],action[2]])
        
        elif self.flap_behavior == 'iso_clapping':
            actions = np.array([action[0],action[0],action[1],action[2]])
                
        elif self.flap_behavior == 'free':
            actions = np.array([action[0], action[1]])
            
        elif self.flap_behavior == 'test':
            actions =  np.array([action[0],action[1],action[2],action[3]]) #top, bottom, right, left
            # actions =  np.array([5,5,5,5])
            # actions =  np.array([20, 20])
        
        elif self.flap_behavior == 'LR_symmetric':
            actions = np.array([action[0],action[1],-action[2],action[2]])  # top, bottom are independent, left and right are symmetric
            
        elif self.flap_behavior == 'LR_only_symmetric':
            actions = np.array([12., 6.6, -action[0], action[0]])
            
        elif self.flap_behavior == 'top_only':
            actions = np.array([action[0], 6.6, 0., 0.])  # Bottom flap at physical zero, top free, LR at zero
            
        elif self.flap_behavior == 'bottom_only':
            actions = np.array([12., action[0], 0., 0.])  # Top flap at physical zero, bottom free, LR at zero
            
        elif self.flap_behavior == 'TB_only':
            actions = np.array([action[0], action[1], 0., 0.])  # top, bottom independent, LR at 0
            
        else:
            raise RuntimeError('flap behavior defined incorrectly')
            

        for num_flaps in range(4):        
            self.previous_action[num_flaps] = actions[num_flaps]
        #initial = time.time()
        
        

        self.env_server.send_payload(
            payload = actions,
            #sending_mask = ">dddd"
            sending_mask = ">dddd"
        )

        
        
        #print(received_data)
        #print('The length of the payload is ', len(received_data))
        
        
        received_data = self._receive_payload()
        #print('The received data is', received_data[0:3])
        
        
        current_pressure = np.zeros(54,) 
        current_angles = np.zeros(4,)
        current_pdiff = np.zeros(28,)
        current_base_pressure = np.zeros(36,)
        
            
        left_indices = [3,4,5,9,10,11,15,16,17,21,22,23,27,28,29,33,34,35]  #left indices of the base
        
        right_indices = [6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38] #right indices of the base
        
        top_indices =  [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        bottom_indices = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
        
        
        
        current_pdiff[0] = np.mean(received_data[left_indices]) - np.mean(received_data[right_indices])
        ud_pdiff = np.mean(received_data[top_indices]) - np.mean(received_data[bottom_indices])
        
        current_pdiff[1:28] = received_data[39:66]
        current_angles = received_data[67:71]
        current_pressure = received_data[self.all_indices]
        current_base_pressure = received_data[3:39]
        


        self.mean_basepressure = np.mean(current_base_pressure)
        self.meanabs_pdiff = current_pdiff[0]
        self.encoder1 = received_data[68]
        self.episodic_bp += self.mean_basepressure
        self.episodic_pdiff += self.meanabs_pdiff
        

         
        self.current_ForceX = received_data[0]
        self.current_ForceY = received_data[1]
        self.current_ForceZ = received_data[2]
        self.total_drag += abs(self.current_ForceX - self.reward_compensation)
        self.total_side_force += abs(self.current_ForceY)
        self.total_lift += abs(self.current_ForceZ)
        
        if self.n_step == self.total_steps:
            if (self.total_drag/self.n_step) < abs(self.avg_baseline_drag[0]):
                #info['indicator'] = True
                self.reward_shaping_trigger = True
                #self.last_epi_trigger = True

        
        # Write latest data to buffer
        self.write_history_parameters()
        
        self.avg_window = min(self.n_step, self.avg_window_max)
        
        if self.reward_function == 'bo_reward':
            self.reward = abs(self.avg_baseline_drag[0]) - abs(self.current_ForceX - self.reward_compensation)
            
        elif self.reward_function == 'bo_reward_avg':
            avg_ForceX = np.mean(self.history_parameters["drag"].get()[-self.avg_window:])
            self.reward = abs(self.avg_baseline_drag[0]) - abs(avg_ForceX - self.reward_compensation)
            
        elif self.reward_function == 'bo_reward_reset_each_episode':
            self.reward = abs(self.calibration_drag) - abs(avg_ForceX)

        elif self.reward_function == 'force_reward':
            self.reward =  abs(self.avg_baseline_drag[0]) - abs(self.current_ForceX - self.reward_compensation) - 2*abs(self.current_ForceY - self.avg_baseline_side[-1])#2*abs(avg_ForceY)
            
        elif self.reward_function == 'avg_force_reward':
            avg_ForceY = np.mean(self.history_parameters["side"].get()[-1:])
            avg_ForceX = np.mean(self.history_parameters["drag"].get()[-self.avg_window:])
            avg_ForceZ = np.mean(self.history_parameters["lift"].get()[-1:])
            
            self.sideforce_penalty = 0.6
            #if self.last_epi_trigger == False:
            if self.twoflapmodes == 'LR':    
                self.reward = abs(self.avg_baseline_drag[0]) - abs(avg_ForceX - self.reward_compensation) - self.sideforce_penalty*abs(avg_ForceY - self.avg_baseline_side[-1])
            elif self.twoflapmodes == 'UD':
                self.reward = abs(self.avg_baseline_drag[0]) - abs(avg_ForceX - self.reward_compensation) - self.sideforce_penalty*abs(avg_ForceZ - self.avg_baseline_lift[-1])
            else:
                
                assert 'The two flapmodes is not defined'
                 
        elif self.reward_function == 'base_pressure_reward':
            
            temporal_abs_shift_pd =  abs(current_pdiff[0] - self.avg_baseline_pdiff[-1])
            
            self.reward = np.mean(self.history_parameters["mean_basepressure"].get()[-self.avg_window:]) - self.reward_compensation 
            - 2 * temporal_abs_shift_pd
            
            
        elif self.reward_function == 'base_pressure_periodic':
            PFscore = 0
            BPscore = np.mean(self.history_parameters["mean_basepressure"].get()[-self.avg_window:])
            if self.n_step >= self.periodic_window:
                actionsequence = self.history_parameters["encoder1"].get()[-self.periodic_window:]
                actionsequence = savgol_filter(actionsequence, 5, 2)
                actionPF = self.compute_PF(actionsequence, 80)
                PFscore = self.similarity_PF(actionPF, 8.9, epsilon=2, p=2)
                self.total_actionPF += actionPF
            
            self.reward = (BPscore - self.reward_compensation ) + 5 * PFscore
            
        elif self.reward_function == 'pdiff_reward':
            
            temporal_abs_shift_pd = - abs(current_pdiff[0] - self.avg_baseline_pdiff[-1])
            instant_change = abs(temporal_abs_shift_pd - self.history_parameters["abs_shift_pd"].get_value(-1))
            
            
            self.history_parameters["abs_shift_pd"].extend(temporal_abs_shift_pd)
            self.history_parameters["instand_pdchange"].extend(instant_change)
            self.reward = np.mean(self.history_parameters["abs_shift_pd"].get()[-1:]) 
            + 0.1 * np.mean(self.history_parameters["mean_basepressure"].get()[-self.avg_window:]) - self.reward_compensation

            
            #print('Part A', np.mean(self.history_parameters["abs_shift_pd"].get()[-self.avg_window:]))
            #print('Part B', 0.05 * np.mean(self.history_parameters["mean_basepressure"].get()[-self.avg_window:]))
            #print('Part C', 0.2 * instant_change)
            
        elif self.reward_function == 'ud_pdiff_reward':
            
            temporal_abs_shift_pd = - abs(ud_pdiff - self.avg_baseline_pdiff[-1])
            
            self.history_parameters["abs_shift_pd"].extend(temporal_abs_shift_pd)
            
            self.reward = np.mean(self.history_parameters["abs_shift_pd"].get()[-1:]) 
            + 0.1 * np.mean(self.history_parameters["mean_basepressure"].get()[-self.avg_window:]) - self.reward_compensation
            
            
        
        
        elif self.reward_function == 'fft_reward':
            
            self.reward = 0
            if self.n_step>200:
                self.reward  = - 20 * self.compute_pwelch(self.history_parameters["pdiff_abs"].get()[-200:], 80)
                #print('Part B', - self.compute_pwelch(self.history_parameters["pdiff_abs"].get()[-200:], 80))
            #    self.compute_pwelch(self.history_parameters["pdiff_abs"].get()[-200:], 80)



        else:
            raise Exception("Undefined reward function.")


        self.total_reward += self.reward

        

        
        if self.pdiff_obs ==True:
            observations = current_pdiff
        else:
            observations = current_pressure
            
            if self.twoflapmodes =='LR':
                observations = np.concatenate((observations, np.array([current_pdiff[0]])))
            elif self.twoflapmodes =='UD':
                observations = np.concatenate((observations, np.array([ud_pdiff])))
            else:
                assert 'The two flap modes is not defined'
                
        if self.include_anlge == True:
            
            if self.action_shape == 2 and self.twoflapmodes == 'UD':
                
                observations = np.concatenate((observations, current_angles[0:2]))
            
            elif self.action_shape == 2 and self.twoflapmodes == 'LR':
                observations = np.concatenate((observations, current_angles[2:4]))
            
            else:
                observations = np.concatenate((observations, current_angles))
            
            
        
        if self.include_action == True:

            if self.action_shape == 2 and self.twoflapmodes == 'UD':
                
                last_action = np.concatenate([self.history_parameters["action_top"].get()[-self.steps_action_augmented:], self.history_parameters["action_bottom"].get()[-self.steps_action_augmented:]])

            elif self.action_shape == 2 and self.twoflapmodes == 'LR':

                last_action = np.concatenate([self.history_parameters["action_right"].get()[-self.steps_action_augmented:], self.history_parameters["action_left"].get()[-self.steps_action_augmented:]])
            
            else:
                
                last_action = np.concatenate([self.history_parameters["action_top"].get()[-self.steps_action_augmented:],self.history_parameters["action_bottom"].get()[-self.steps_action_augmented:], 
                                                 self.history_parameters["action_right"].get()[-self.steps_action_augmented:], self.history_parameters["action_left"].get()[-self.steps_action_augmented:]])
                
            
            observations = np.concatenate([observations, last_action])

        #print(observations)
        
            
        return observations, self.reward, self.done, info

    def reset(self) -> np.ndarray:
        
        self.counter = 0

        self.done = False
        self.previous_pressure = {}
        self.previous_ForceX = None
        self.previous_ForceY= None
        self.previous_ForceZ = None
        self.previous_power = {}
        self.previous_voltage = {}
        self.previous_action = {}
        self.previous_angles = {}
        self.n_step = 0
        

        self.n_episodes += 1
        self._logger.info(f"Episode Number: {self.n_episodes}")
        
        
        if self.n_episodes == 0:
            self.baseline_duras = 1000
        else:
            self.baseline_duras = 1000
        
        # Buffer for forces
        self.history_parameters["drag"] = RingBuffer(self.size_history)
        self.history_parameters["lift"] = RingBuffer(self.size_history)
        self.history_parameters["side"] = RingBuffer(self.size_history)
        self.history_parameters["abs_shift_pd"] = RingBuffer(self.size_history)
        self.history_parameters["instand_pdchange"] = RingBuffer(self.size_history)
        self.history_parameters["mean_basepressure"] = RingBuffer(self.size_history)
        self.history_parameters["pdiff_abs"] = RingBuffer(self.size_history)
        self.history_parameters["encoder1"] = RingBuffer(self.total_steps)
        self.history_parameters["action_top"] = RingBuffer(self.size_history)
        self.history_parameters["action_bottom"] = RingBuffer(self.size_history)
        self.history_parameters["action_right"] = RingBuffer(self.size_history)
        self.history_parameters["action_left"] = RingBuffer(self.size_history)
        
        
        self.history_parameters["baseline_basepressure"] = RingBuffer(self.baseline_duras)
        self.history_parameters["baseline_pdiff"] = RingBuffer(self.baseline_duras)
        self.history_parameters["baseline_drag"] = RingBuffer(self.baseline_duras)
        self.history_parameters["baseline_side"] = RingBuffer(self.baseline_duras)
        self.history_parameters["baseline_lift"] = RingBuffer(self.baseline_duras)
        


        self._logger.warning("ENV RESET")
        
        
        #self.env_server.send_payload([0, 0, 0, 0], sending_mask=">dddd") # 
        self.env_server.send_payload([0, 0, 0, 0], sending_mask=">dddd") # 
        #print('#indicator')

        time.sleep(6)
        
        if self.n_episodes % self.interval_record_baseline == 0:

            self._logger.info(f"Obtain Baseline")
            time.sleep(0.25)
            self.baseline_bp = 0
            print('#entering baseline recording indicator')
            
            for i in range(self.baseline_duras):
                baseline_data = self._receive_payload()
                
                time.sleep(0.0125)
                
                left_indices = [3,4,5,9,10,11,15,16,17,21,22,23,27,28,29,33,34,35]  #left indices of the base
                right_indices = [6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38] #right indices of the base
                top_indices =  [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                bottom_indices = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
                
                baseline_base_pressure = np.zeros(36,)
                baseline_base_pressure = baseline_data[3:39]
                
                if self.twoflapmodes == 'LR':
                    self.baseline_pdiff = np.mean(baseline_data[left_indices]) - np.mean(baseline_data[right_indices])
                else:
                    self.baseline_pdiff = np.mean(baseline_data[top_indices]) - np.mean(baseline_data[bottom_indices])
                    
                    
                self.mean_baseline_basepressure = np.mean(baseline_base_pressure)
                self.baseline_bp += self.mean_baseline_basepressure
                
                self.history_parameters["baseline_basepressure"].extend(self.mean_baseline_basepressure)
                self.history_parameters["baseline_pdiff"].extend(self.baseline_pdiff)
                #self.history_parameters["pdiff_abs"].extend(self.baseline_pdiff)
                self.history_parameters["baseline_drag"].extend(baseline_data[0])
                self.history_parameters["baseline_side"].extend(baseline_data[1])
                self.history_parameters["baseline_lift"].extend(baseline_data[2])
                
                
            self.avg_baseline_pressure = np.append(self.avg_baseline_pressure, np.mean(self.history_parameters["baseline_basepressure"].get()))
            self.avg_baseline_pdiff = np.append(self.avg_baseline_pdiff, np.mean(self.history_parameters["baseline_pdiff"].get()))
            self.avg_baseline_drag = np.append(self.avg_baseline_drag, np.mean(self.history_parameters["baseline_drag"].get()))
            self.avg_baseline_side = np.append(self.avg_baseline_side, np.mean(self.history_parameters["baseline_side"].get()))
            self.avg_baseline_lift = np.append(self.avg_baseline_lift, np.mean(self.history_parameters["baseline_lift"].get()))
            
            
            
            
                           
            if self.avg_baseline_pressure.size > 1:
                
                self._logger.info(f"Compensate rewards")
                if self.reward_function == 'avg_force_reward' or 'bo_reward_avg' or 'bo_reward':
                    self.reward_compensation = self.avg_baseline_drag[-1] - self.avg_baseline_drag[0]
                    print("Using drag force to compute reward compensation. Force drifting", self.reward_compensation)
                
                elif self.reward_function == 'bo_reward_reset_each_episode':
                    self.calibration_drag = self.avg_baseline_drag[-1]
                
                else:
                    self.reward_compensation = self.avg_baseline_pressure[-1] - self.avg_baseline_pressure[0]
                    print("Using pressure to compute reward compensation. Pressure drifting", self.reward_compensation)
                
                
                with open(f"_whitemamba_drift_record.txt", "a") as WriteCompensation:
                    WriteCompensation.write(str(self.n_episodes)+"\t"+ str(self.reward_compensation)+"\t"+ str(self.avg_baseline_pressure[-1])+"\t"+str(self.avg_baseline_pdiff[-1]))
                    WriteCompensation.write("\n")
            else:
                with open(f"_whitemamba_drift_record.txt", "a") as WriteCompensation:
                    WriteCompensation.write(str(self.n_episodes)+"\t"+ str(self.avg_baseline_pressure[-1]))
                    WriteCompensation.write("\n")
                
        
        filename = '_whitemamba_reward.txt'
        file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
        with open(f"_whitemamba_reward.txt", "a") as WriteReward:
            
            if not file_exists:
                WriteReward.write('Episodes'+"\t" + 'Total Reward' + "\t" + 'Episodeic Base Pressure'
                                  + "\t"+ 'Episodic Pdiff' + "\t"+ 'Average Drag' + "\t"+ 'Average Side Force' + "\t"+ 'Average Lift')
                
                WriteReward.write("\n")
                
            if self.reward_shaping_trigger == True:
                
                WriteReward.write(str(self.n_episodes)+"\t" + str(self.total_reward) + "\t" + str(self.episodic_bp) 
                                  + "\t"+ str(self.episodic_pdiff) + "\t"+ str(self.total_drag/self.total_steps) + "\t"+ str(self.total_side_force/self.total_steps) 
                                  + "\t"+ str(self.total_lift/self.total_steps) + "\t" + "triggered_reward_shaping" )
                
            else:
                
                WriteReward.write(str(self.n_episodes)+"\t" + str(self.total_reward) + "\t" + str(self.episodic_bp) 
                                  + "\t"+ str(self.episodic_pdiff) + "\t"+ str(self.total_drag/self.total_steps) + "\t" + str(self.total_side_force/self.total_steps) + "\t"+ str(self.total_lift/self.total_steps))
                
            WriteReward.write("\n")
            
            

        self.total_reward = 0
        self.episodic_bp = 0
        self.episodic_pdiff = 0
        self.total_actionPF = 0
        self.total_drag = 0
        self.total_side_force = 0
        self.total_lift = 0

    

        #payload = self.env_server.receive_payload(">dddddddddddddddddd")
        #received_data = self.format_payload(payload)
        
        
        received_data = self._receive_payload()
        
        

        self._logger.warning("ENV RESET DONE")


        current_actions = np.zeros(4,)
        current_pressure = np.zeros(54,) 
        current_angles = np.zeros(4,)
        current_pdiff = np.zeros(28,)
        current_base_pressure = np.zeros(36,)
        
            
        left_indices = [3,4,5,9,10,11,15,16,17,21,22,23,27,28,29,33,34,35]  #left indices of the base
        right_indices = [6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38] #right indices of the base
        top_indices =  [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        bottom_indices = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
        
        current_pdiff[0] = np.mean(received_data[left_indices]) - np.mean(received_data[right_indices])
        ud_pdiff = np.mean(received_data[top_indices]) - np.mean(received_data[bottom_indices])
        current_pdiff[1:28] = received_data[39:66]
        current_angles = received_data[67:71]
        current_pressure = received_data[self.all_indices]
        current_base_pressure = received_data[3:39]
            

        current_ForceX = received_data[0]
        current_ForceY = received_data[1]
        current_ForceZ = received_data[2]
        
        self.mean_basepressure = np.mean(current_base_pressure)
        self.meanabs_pdiff = np.absolute(current_pdiff[0])
        self.encoder1 = received_data[68]
        
        actions =  np.array([0,0,0,0]) #top, bottom, right, left
        for num_flaps in range(4):        
            self.previous_action[num_flaps] = actions[num_flaps]
            
        # Write latest data to buffer
        self.write_history_parameters()
        
            
        if self.pdiff_obs == True:
            observations = current_pdiff
        else:
            observations = current_pressure
            
            if self.twoflapmodes =='LR':
                observations = np.concatenate((observations, np.array([current_pdiff[0]])))
            elif self.twoflapmodes =='UD':
                observations = np.concatenate((observations, np.array([ud_pdiff])))
            else:
                assert 'The two flap modes is not defined'
                
        if self.include_anlge == True:
            
            if self.action_shape == 2 and self.twoflapmodes == 'UD':
                
                observations = np.concatenate((observations, current_angles[0:2]))
            
            elif self.action_shape == 2 and self.twoflapmodes == 'LR':
                observations = np.concatenate((observations, current_angles[2:4]))
            
            else:
                observations = np.concatenate((observations, current_angles))
            
        if self.include_action == True:

            if self.action_shape == 2 and self.twoflapmodes == 'UD':
                                
                last_action = np.concatenate([self.history_parameters["action_top"].get()[-self.steps_action_augmented:], self.history_parameters["action_bottom"].get()[-self.steps_action_augmented:]])

                        
            elif self.action_shape == 2 and self.twoflapmodes == 'LR':
                
                last_action = np.concatenate([self.history_parameters["action_right"].get()[-self.steps_action_augmented:], self.history_parameters["action_left"].get()[-self.steps_action_augmented:]])
            
            else:
                
                last_action = np.concatenate([self.history_parameters["action_top"].get()[-self.steps_action_augmented:],self.history_parameters["action_bottom"].get()[-self.steps_action_augmented:], 
                                                 self.history_parameters["action_right"].get()[-self.steps_action_augmented:], self.history_parameters["action_left"].get()[-self.steps_action_augmented:]])
                
            
            observations = np.concatenate([observations, last_action])
            
        
        return observations
        
        

    def format_payload(self, payload) -> np.ndarray:
        """
        Formatting payload which is going to be sent to training agent
        """
        formatted_payload = np.array(payload, dtype=np.float64)

        return formatted_payload
    
    
    def compute_PF(self, input_sequence, sampling_rate):
        
        input_sequence = detrend(input_sequence)
        n = len(input_sequence)
        hanning_window = np.hanning(n)
        input_sequence = input_sequence * hanning_window
        
        fft_result = np.fft.fft(input_sequence)
        frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
        magnitude_spectrum = np.abs(fft_result)[:n//2]
        positive_frequencies = frequencies[:n//2]
        peak_index = np.argmax(magnitude_spectrum)
        peak_frequency = positive_frequencies[peak_index]
        plt.semilogx(positive_frequencies, magnitude_spectrum*positive_frequencies)
        plt.grid(True)
        plt.show()
        return peak_frequency
    
    def compute_pwelch(self, input_sequence, sampling_rate):
        # Detrend the input sequence
        input_sequence = detrend(input_sequence)
        
        # Calculate the power spectral density using the Welch method
        #frequencies, power_spectrum = welch(input_sequence, fs=sampling_rate, window='hann', nperseg=256, noverlap=None, detrend='constant', scaling='spectrum')
        nwindow = 2^6
        frequencies, power_spectrum = welch(input_sequence, fs=sampling_rate, window='hann', detrend =False)
        # Find the peak in the power spectrum
        freq_range_indices = np.where((frequencies >= 8) & (frequencies <= 10.5))[0]
        power_sum = np.sum(power_spectrum[freq_range_indices])

        #plt.figure(figsize=(10, 6))
        #plt.plot([10.5,10.5], [0,0.0025])
        #plt.plot([8,8], [0,0.0025])
        #plt.semilogx(frequencies, power_spectrum*frequencies)
        #plt.title('Power Spectral Density')
        #plt.xlabel('Frequency [Hz]')
        #plt.ylabel('Power/Frequency [dB/Hz]')
        #plt.xlim([0.1, 40])
        #plt.ylim([0, 0.003])
        #plt.grid(True)
        #plt.show()
    
        return power_sum
    
    def similarity_PF(self, x, y, epsilon=2, p=2):
        return 1 / (abs(x-y) + epsilon) ** p
    

