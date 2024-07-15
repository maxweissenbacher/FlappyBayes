import logging
import numpy as np

from gym import spaces
from DrlPlatform import PlatformEnvironment
import time
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

    def __init__(self):

        super(AhmedBody_AllObservations, self).__init__()
        
        self.size_history = 20000 # Size of the RingBuffer
        self.nstack = 160 # The number of framestack
        self.nskip = 1 #The number of frame skip
        self.obs_with_force = False # Whether include drag force in observations
        self.include_anlge = False # Whether include angle in observations
        self.Use_FS = True # Whether use frame stack
        self.useKF = False # Whether use KF
        self.flap_behavior = 'iso_snaking' # Whether use snaking, clapping, free
        self.interval_record_baseline = 2 # Steps interval to extend reset time and obtain forces in baseline, which measures load cell drifting
        self.baseline_duras = 100  # Calculate baseline for 0.02 * baseline_duras seconds
        self.reward_function = 'force_reward'
        
        # Create ring buffers to hold recent history of jet values, probe values, lift, drag, area
        self.history_parameters = {}
        
        # Variables store forces drifting from baseline flow, and compensation for reward functions
        self.avg_baseline_drag = np.array([])
        self.avg_baseline_lift = np.array([])
        self.avg_baseline_side = np.array([])
        self.reward_compensation = 0
        
        # Buffer for pressure measurements
        for num_sensors in range(7):
            self.history_parameters["sensor_{}".format(num_sensors)] = RingBuffer(self.size_history)
        
        # Buffer for power and actions (voltage)
        for num_flaps in range(4):
            self.history_parameters["voltage_{}".format(num_flaps)] = RingBuffer(self.size_history)
            self.history_parameters["power_{}".format(num_flaps)] = RingBuffer(self.size_history)
            self.history_parameters["action_{}".format(num_flaps)] = RingBuffer(self.size_history)
            
        # Buffer for forces
        self.history_parameters["drag"] = RingBuffer(self.size_history)
        self.history_parameters["lift"] = RingBuffer(self.size_history)
        self.history_parameters["side"] = RingBuffer(self.size_history)
        
        self._logger = logging.getLogger(__name__)
        self.pressure_max = np.finfo(np.float32).max
        self.ForceX_max = np.finfo(np.float32).max
        self.ForceY_max = np.finfo(np.float32).max
        self.ForceZ_max = np.finfo(np.float32).max
        self.power_max = np.finfo(np.float32).max
        self.voltage_max = np.finfo(np.float32).max
        self.action_max = np.float32(5)
        
        limit_obs = np.array([])
        for nums in range(self.nstack * 11):
            limit_obs = np.append(limit_obs, self.pressure_max)

        if self.flap_behavior == 'snaking' or self.flap_behavior == 'clapping':
            self.action_shape = 2
            
        elif self.flap_behavior == 'iso_snaking' or self.flap_behavior == 'iso_clapping':
            self.action_shape = 3
            
        elif self.flap_behavior == 'free':
            self.action_shape = 4

        elif self.flap_behavior == 'test':
            self.action_shape = 3
        else:
            assert 'Flap behavior is not defiend correctly'
            
        limit_act = np.array([])
        for nums in range(self.action_shape):
            limit_act = np.append(limit_act, self.action_max)

        n_dim_state = 11 # 7 pressures + 4 actions
        n_dim_obs = 11

        # Initialize the Kalman Filter

        if self.useKF:
            transition_matrices = np.eye(n_dim_state)  # You might need to modify this based on your system's dynamics
            observation_matrices = np.eye(n_dim_obs)  # Assuming you observe all state variables directly
            self.kf = KalmanFilter(transition_matrices, observation_matrices)

        self.observation_space = spaces.Box(-limit_obs, limit_obs, dtype=np.float32)  # type: ignore
        self.action_space = spaces.Box(-limit_act, limit_act, dtype=np.float32)
        self.total_reward = 0.0
        self.total_drag = 0.0
        self.done = False
        self.n_episodes = -1
        self.prev_time = 0
        
        # Historical data
        self.previous_pressure = {}
        self.previous_ForceX = None
        self.previous_ForceY= None
        self.previous_ForceZ = None
        self.previous_power = {}
        self.previous_voltage = {}
        self.previous_action = {}
        # self.previous_pressure_1 = None
        # self.previous_pressure_2 = None
        # self.previous_pressure_3 = None
        # self.previous_pressure_4 = None
        # self.previous_pressure_6 = None
        # self.previous_pressure_7 = None
        # self.previous_pressure_8 = None
        # self.previous_power_1 = None
        # self.previous_power_2 = None
        # self.previous_power_3 = None
        # self.previous_power_4 = None
        # self.previous_voltage_1 = None
        # self.previous_voltage_2 = None
        # self.previous_voltage_3 = None
        # self.previous_voltage_4 = None
        # self.previous_action_1 = None
        # self.previous_action_2 = None
        # self.previous_action_3 = None
        # self.previous_action_4 = None
        
    def write_history_parameters(self):
        '''
        Add data of last step to history buffers
        :return:
        '''
        
        # Buffer for pressure measurements
        for num_sensors in range(7):
            self.history_parameters["sensor_{}".format(num_sensors)].extend(self.previous_pressure[num_sensors])
        
        # Buffer for flap angles and actions (voltage)
        for num_flaps in range(4):
            # self.history_parameters["angles_{}".format(num_flaps)].extend(self.previous_angles[num_flaps])
            self.history_parameters["voltage_{}".format(num_flaps)].extend(self.previous_voltage[num_flaps])
            self.history_parameters["power_{}".format(num_flaps)].extend(self.previous_power[num_flaps])
            self.history_parameters["action_{}".format(num_flaps)].extend(self.previous_action[num_flaps])

        self.history_parameters["drag"].extend(np.array(self.previous_ForceX))
        self.history_parameters["lift"].extend(np.array(self.previous_ForceZ))
        self.history_parameters["side"].extend(np.array(self.previous_ForceY))
        
    def _receive_payload(self):

        # Returns history_lift, history_drag, current_lift, current_drag

        payload = self.env_server.receive_payload(">dddddddddddddddddd")
        formatted_payload = self.format_payload(payload)
        
        # current_pressure_1, current_pressure_2, current_pressure_3,\
        #     current_pressure_4, current_pressure_6, current_pressure_7, current_pressure_8,\
        #     current_ForceX, current_ForceY, current_ForceZ, current_power_1, current_power_2,\
        #     current_power_3, current_power_4, current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4= \
        #     formatted_payload[0], formatted_payload[1], formatted_payload[2], formatted_payload[3], formatted_payload[4], \
        #         formatted_payload[5], formatted_payload[6], formatted_payload[7], formatted_payload[8], formatted_payload[9], \
        #         formatted_payload[10], formatted_payload[11], formatted_payload[12], formatted_payload[13], \
        #         formatted_payload[14], formatted_payload[15], formatted_payload[16], formatted_payload[17]

        for num_sensors in range(7):
            self.previous_pressure[num_sensors] = formatted_payload[num_sensors]
        
        self.previous_ForceX = formatted_payload[7]
        self.previous_ForceY = formatted_payload[8]
        self.previous_ForceZ = formatted_payload[9]
        
        for num_flaps in range(4):
            # self.history_parameters["angles_{}".format(num_flaps)] = formatted_payload[num_flaps+10]
            self.previous_power[num_flaps] = formatted_payload[num_flaps+10]
            self.previous_voltage[num_flaps] = formatted_payload[num_flaps+14]
            
        
        
        # self.previous_pressure_1 = current_pressure_1
        # self.previous_pressure_2 = current_pressure_2
        # self.previous_pressure_3 = current_pressure_3
        # self.previous_pressure_4 = current_pressure_4
        # self.previous_pressure_6 = current_pressure_6
        # self.previous_pressure_7 = current_pressure_7
        # self.previous_pressure_8 = current_pressure_8
        # self.previous_ForceX = current_ForceX
        # self.previous_ForceY = current_ForceY
        # self.previous_ForceZ = current_ForceZ
        # self.previous_power_1 = current_power_1
        # self.previous_power_2 = current_power_2
        # self.previous_power_3 = current_power_3
        # self.previous_power_4 = current_power_4
        # self.previous_voltage_1 = current_voltage_1
        # self.previous_voltage_2 = current_voltage_2
        # self.previous_voltage_3 = current_voltage_3
        # self.previous_voltage_4 = current_voltage_4

        return formatted_payload
    
    # current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4,\
    #         current_pressure_6, current_pressure_7, current_pressure_8, current_ForceX, current_ForceY, \
    #         current_ForceZ, current_power_1, current_power_2, current_power_3, current_power_4, \
    #         current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4
    
    def Obs_FS(self, N_stack, N_skip, probe_nums = 7, action_nums = 4):
        
        # Function implementing framestack with frame skipping.
        # This helps extend the horizon into the past with a relatively high frequency of sampling frames.
        observations = np.array([])
        buffer_required = N_stack * N_skip;
        if buffer_required > self.size_history:
            raise Exception("Buffer size too small for FrameStack.")
        
        past_pressure = np.zeros(probe_nums * N_stack)
        past_voltage = np.zeros(action_nums * N_stack)
        past_angle = np.zeros(action_nums * N_stack)
        past_drag = np.zeros(N_stack)
        past_lift = np.zeros(N_stack)
            
        for i in range(N_stack) :
            
            if N_skip ==0:
                assert 'invalid value for N_skip'
            index_value = -1 if i == 0 else -i * N_skip -1 #if i = 0, take the current obseravtion
            
            
            
            # Obtain past pressure measurements
            past_pressure[i*probe_nums : i*probe_nums + probe_nums] = [self.history_parameters["sensor_{}".format(num_sensors)].get_value(index_value)
                                   for num_sensors in range(probe_nums)]

            past_voltage[i*action_nums : i*action_nums + action_nums] = [self.history_parameters["voltage_{}".format(num_actions)].get_value(index_value)
                                   for num_actions in range(action_nums)]
                                   
            
            past_angle[i*action_nums : i*action_nums + action_nums] = [self.history_parameters["action_{}".format(num_actions)].get_value(index_value)
                                    for num_actions in range(action_nums)]
                                    
            past_drag[i*1 : i*1 + 1] = [self.history_parameters["drag"].get_value(index_value)]
            past_lift[i*1 : i*1 + 1] = [self.history_parameters["lift"].get_value(index_value)]
            
            
            #print(index_value)

            #print('check action buffer')
            
            #print([self.history_parameters["jet_{}".format(num_actions)].get_value(index_value)
            #                       for num_actions in range(action_nums)])
          

            # Obtain past voltage (or actions) using list comprehension
            #past_voltage[i*action_nums : i*action_nums + action_nums] = [self.history_parameters["voltage_{}".format(num_flaps)].get_value(index_value)
                                          #for num_flaps in range(action_nums)]
            # Form into a past frame and append to observations

        #np.set_printoptions(threshold=np.inf)
        #print(past_action)
        #print('past action is above me')
        #print(past_pressure)
        if self.include_anlge == True:
                observations = np.concatenate((past_pressure,  past_voltage, past_angle))
        else:
                observations = np.concatenate((past_pressure,  past_voltage))
        
        if self.obs_with_force == True:
                observations = np.concatenate(observations, past_drag, past_lift)
               
                
        
        #print(observations)
        #print(len(observations))
        return observations
    
    def step(self, action):
        # Negative reward for a step
        info = {}
        Ahmed_section = 0.216 * 0.160
        Velocity = 15
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
            actions = action

        elif self.flap_behavior == 'test':
            actions =  np.array([action[2], -action[2] ,0, 0])
        
        else:
            assert 'flap behavior defiend incorrectly'
            

        for num_flaps in range(4):        
            self.previous_action[num_flaps] = actions[num_flaps]
        #initial = time.time()

        self.env_server.send_payload(
            payload = actions,
            sending_mask = ">dddd"
        )

        #sent = time.time()
        #print("Sent Elapsed: ", (sent - start))
        #loop_time = sent - self.prev_time
        #print("Loop time: ", loop_time)
        #self.prev_time = sent

        received_data = self._receive_payload()
        
        # Write latest data to buffer
        self.write_history_parameters()

        current_pressure = np.zeros(7,)
        current_power = np.zeros(4,)
        current_voltage = np.zeros(4,)
        current_angle = np.zeros(4,)
        
        for num_sensors in range(7):
            current_pressure[num_sensors] = received_data[num_sensors]
        
        current_ForceX = received_data[7]
        current_ForceY = received_data[8]
        current_ForceZ = received_data[9]
        
        for num_flaps in range(4):
            # self.history_parameters["angles_{}".format(num_flaps)] = formatted_payload[num_flaps+10]
            current_power[num_flaps] = received_data[num_flaps+10]
            current_voltage[num_flaps] = received_data[num_flaps+14]
            # current_angle[num_flaps] =
        # current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4, \
        #     current_pressure_6, current_pressure_7, current_pressure_8, current_ForceX, current_ForceY, \
        #     current_ForceZ, current_power_1, current_power_2, current_power_3, current_power_4, \
        #     current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4 = self._receive_payload()

        #receive = time.time()
        #print("Sent Elapsed: ", (sent - start))
        #loop_time = receive - self.prev_time
        #print("Loop time: ", loop_time)
        #self.prev_time = receive

        #receive = time.time()
        #print("Received Elapsed: ", (receive - start))
        #print("Time to do the entire Labview loop: ", (receive - sent))
        # Compute episodic reward (as a function of CL and CD)
        #self.reward = (current_pressure_1 + current_pressure_2 + current_pressure_3 + current_pressure_4 + \
         #current_pressure_6 + current_pressure_7 + current_pressure_8) * Area / 7
        if self.reward_function == 'force_reward':
            self.reward =  abs(self.avg_baseline_drag[0]) - abs(current_ForceX - self.reward_compensation) - 0.2*current_ForceY
        else:
            self.reward = - np.sum(current_power) + (abs(self.avg_baseline_drag[0]) - abs(current_ForceX-self.reward_compensation)) * Velocity

        #self.reward = self.reward = -0.2 * self._lift_reward(current_ForceZ) - self._drag_reward(current_ForceX)
        self.total_reward += self.reward
        self.total_drag += abs(current_ForceX-self.reward_compensation)
        
        # TODO: ADD FrameStack
        
            
        if self.Use_FS == True:
            observations = self.Obs_FS(self.nstack, self.nskip)
            
        else:
        
            if self.include_anlge == True:
                observations = np.append(current_pressure, current_voltage, current_angle)
            
            else:
                observations = np.append(current_pressure, current_voltage)
                
            
            if self.obs_with_force:
                observations = np.append(observations, current_ForceX)
        # observations = np.array([ current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4,\
        #     current_pressure_6, current_pressure_7, current_pressure_8, \
        #     current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4],
        #                              dtype=np.float32)
            
        # Run KF and update observations
        # if self.useKF:
        #     self.kf = self.kf.em(measurements, n_iter=5)
        #     (filtered_state_means, filtered_state_covariances) = self.kf.filter(measurements)
        #     observations = filtered_state_means[-1]
        #     next_state = np.append(next_state, action)
        
        self._logger.info(f"Reward: {self.reward}")

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
        
        for num_sensors in range(7):
            self.history_parameters["sensor_{}".format(num_sensors)] = RingBuffer(self.size_history)
        
        # Buffer for power and actions (voltage)
        for num_flaps in range(4):
            self.history_parameters["voltage_{}".format(num_flaps)] = RingBuffer(self.size_history)
            self.history_parameters["power_{}".format(num_flaps)] = RingBuffer(self.size_history)
            self.history_parameters["action_{}".format(num_flaps)] = RingBuffer(self.size_history)
            
        # Buffer for forces
        self.history_parameters["drag"] = RingBuffer(self.size_history)
        self.history_parameters["lift"] = RingBuffer(self.size_history)
        self.history_parameters["side"] = RingBuffer(self.size_history)
        
        self.history_parameters["baseline_drag"] = RingBuffer(self.baseline_duras)
        self.history_parameters["baseline_lift"] = RingBuffer(self.baseline_duras)
        self.history_parameters["baseline_side"] = RingBuffer(self.baseline_duras)
        
        
        # self.previous_pressure_1 = None
        # self.previous_pressure_2 = None
        # self.previous_pressure_3 = None
        # self.previous_pressure_4 = None
        # self.previous_pressure_6 = None
        # self.previous_pressure_7 = None
        # self.previous_pressure_8 = None
        # self.previous_ForceX = None
        # self.previous_ForceY= None
        # self.previous_ForceZ = None
        # self.previous_power_1 = None
        # self.previous_power_2 = None
        # self.previous_power_3 = None
        # self.previous_power_4 = None
        # self.previous_voltage_1 = None
        # self.previous_voltage_2 = None
        # self.previous_voltage_3 = None
        # self.previous_voltage_4 = None

        self._logger.warning("ENV RESET")
        
        # if not self.previous_action_1 and not self.previous_action_2 and not self.previous_action_3 and not self.previous_action_4 :
        #     self.previous_action_1 = 0
        #     self.previous_action_2 = 0
        #     self.previous_action_3 = 0
        #     self.previous_action_4 = 0
        
        self.env_server.send_payload([0, 0, 0, 0], sending_mask=">dddd") # 
        
        for num_flaps in range(4):        
            self.previous_action[num_flaps] = np.float32(0)
            
        # self.previous_action_1 = None
        # self.previous_action_2 = None
        # self.previous_action_3 = None
        # self.previous_action_4 = None

        self.n_episodes += 1
        self._logger.info(f"Episode Number: {self.n_episodes}")
        
        if self.n_episodes % self.interval_record_baseline == 0:
            self._logger.info(f"Obtain Baseline")
            for i in range(self.baseline_duras):
                baseline_data = self._receive_payload()
                time.sleep(0.02)
                self.history_parameters["baseline_drag"].extend(baseline_data[8])
                self.history_parameters["baseline_lift"].extend(baseline_data[10])
                self.history_parameters["baseline_side"].extend(baseline_data[9])
                
            self.avg_baseline_drag = np.append(self.avg_baseline_drag, np.mean(self.history_parameters["baseline_drag"].get()))
            self.avg_baseline_lift = np.append(self.avg_baseline_lift, np.mean(self.history_parameters["baseline_lift"].get()))
            self.avg_baseline_side = np.append(self.avg_baseline_side, np.mean(self.history_parameters["baseline_side"].get()))
            
            if self.avg_baseline_drag.size > 1:
                self._logger.info(f"Compensate rewards")
                self.reward_compensation = self.avg_baseline_drag[-1] - self.avg_baseline_drag[0]
                print("Force drifting", self.reward_compensation)
                with open(f"drift_record.txt", "a") as WriteCompensation:
                    WriteCompensation.write(str(self.reward_compensation))
                    WriteCompensation.write("\n")

        with open(f"reward.txt", "a") as WriteReward:
            WriteReward.write(str(self.n_episodes)+"\t"+str(self.total_reward)+"\t"+str(self.total_drag))
            WriteReward.write("\n")

        self.total_reward = 0
        self.total_drag = 0

        #payload = self.env_server.receive_payload(">dddddddddddddddddd")
        #received_data = self.format_payload(payload)
        #time.sleep(1)
        received_data = self._receive_payload()
        
        # Write latest data to buffer
        self.write_history_parameters()

        self._logger.warning("ENV RESET DONE")

        # current_pressure_1, current_pressure_2, current_pressure_3, \
        #     current_pressure_4, current_pressure_6, current_pressure_7, current_pressure_8, \
        #     current_ForceX, current_ForceY, current_ForceZ, current_power_1, current_power_2, \
        #     current_power_3, current_power_4, current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4 = \
        #     formatted_payload[0], formatted_payload[1], formatted_payload[2], formatted_payload[3], formatted_payload[4], \
        #         formatted_payload[5], formatted_payload[6], formatted_payload[7], formatted_payload[8], formatted_payload[9], \
        #         formatted_payload[10], formatted_payload[11], formatted_payload[12], formatted_payload[13], \
        #         formatted_payload[14], formatted_payload[15], formatted_payload[16], formatted_payload[17]

        current_pressure = np.zeros(7,)
        current_power = np.zeros(4,)
        current_voltage = np.zeros(4,)
        
        for num_sensors in range(7):
            current_pressure[num_sensors] = received_data[num_sensors]
        
        current_ForceX = received_data[7]
        current_ForceY = received_data[8]
        current_ForceZ = received_data[9]
        
        for num_flaps in range(4):
            # self.history_parameters["angles_{}".format(num_flaps)] = formatted_payload[num_flaps+10]
            current_power[num_flaps] = received_data[num_flaps+10]
            current_voltage[num_flaps] = received_data[num_flaps+14]
            
        if self.Use_FS == True:
            observations = self.Obs_FS(self.nstack, self.nskip)
            
        else:
        
            if self.include_anlge == True:
                observations = np.append(current_pressure, current_voltage, current_angle)
            
            else:
                observations = np.append(current_pressure, current_voltage)
                
            
            if self.obs_with_force:
                observations = np.append(observations, current_ForceX)
        
        # observations = np.array([current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4,\
        #     current_pressure_6, current_pressure_7, current_pressure_8, \
        #     current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4],
        #                              dtype=np.float32)
        return observations
        
        
    

    def format_payload(self, payload) -> np.ndarray:
        """
        Formatting payload which is going to be sent to training agent
        """
        formatted_payload = np.array(payload, dtype=np.float32)

        # current_pressure_1, current_pressure_2, current_pressure_3, \
        #     current_pressure_4, current_pressure_6, current_pressure_7, current_pressure_8, \
        #     current_ForceX, current_ForceY, current_ForceZ, current_power_1, current_power_2, \
        #     current_power_3, current_power_4, current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4 = \
        #     formatted_payload[0], formatted_payload[1], formatted_payload[2], formatted_payload[3], formatted_payload[4], \
        #         formatted_payload[5], formatted_payload[6], formatted_payload[7], formatted_payload[8], formatted_payload[9], \
        #         formatted_payload[10], formatted_payload[11], formatted_payload[12], formatted_payload[13], \
        #         formatted_payload[14], formatted_payload[15], formatted_payload[16], formatted_payload[17]

        return formatted_payload
    
    # np.array([current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4,\
    #         current_pressure_6, current_pressure_7, current_pressure_8, current_ForceX, current_ForceY, current_ForceZ, \
    #              current_power_1, current_power_2, current_power_3, current_power_4, \
    #                      current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4], dtype=np.float32)

