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

    def __init__(self):

        super(AhmedBody_AllObservations, self).__init__()
        
        self.size_history = 20000 # Size of the RingBuffer

        self.total_steps = int(2048)

        self.obs_with_force = False # Whether include drag force in observations
        self.include_anlge = True # Whether include angle in observations
        self.include_action = True
        self.pdiff_obs = False
        self.append_pdiff = True



        self.baseline_bp = 0
        self.baseline_pdiff = 0
        self.encoder1=0
        self.episodic_bp = 0
        self.episodic_pdiff = 0
        self.target_frequency = 0
        self.total_actionPF = 0
        self.periodic_window = 200
        
       
        
        self.flap_behavior = 'test' # Whether use snaking, clapping, free
        self.interval_record_baseline = 2 # Steps interval to extend reset time and obtain forces in baseline, which measures load cell drifting
        #self.baseline_duras = 500  # Calculate baseline for 0.02 * baseline_duras seconds
        self.reward_function = 'pdiff_reward' #can also use 'force_reward' 'base_pressure_reward'
        
        
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
        
        
        self._logger = logging.getLogger(__name__)
        self.pressure_max = np.finfo(np.float64).max
        self.ForceX_max = np.finfo(np.float64).max
        self.ForceY_max = np.finfo(np.float64).max
        self.ForceZ_max = np.finfo(np.float64).max
        self.power_max = np.finfo(np.float64).max
        self.voltage_max = np.finfo(np.float64).max
        self.action_max = np.float64(4.5)

        
        if self.pdiff_obs == True:
            self.state_shape = 28
        else:
            self.state_shape = 65
        
        
        if self.include_anlge == True:
            self.state_shape += 2  
            

        
        if self.obs_with_force == True:
            self.state_shape += 2
            
            
        

        if self.flap_behavior == 'snaking' or self.flap_behavior == 'clapping':
            self.action_shape = 2
            
        elif self.flap_behavior == 'iso_snaking' or self.flap_behavior == 'iso_clapping':
            self.action_shape = 3
            
        elif self.flap_behavior == 'free':
            self.action_shape = 4

        elif self.flap_behavior == 'test':
            self.action_shape = 1
        else:
            assert 'Flap behavior is not defiend correctly'
            
       

        #n_dim_state = 11 # 7 pressures + 4 actions
        #n_dim_obs = 11

        if self.include_action == True:
            self.state_shape +=  self.action_shape
            
        limit_obs = np.array([])
        
        for nums in range(self.state_shape):
            limit_obs = np.append(limit_obs, self.pressure_max)
            
        limit_act = np.array([])
        for nums in range(self.action_shape):
            limit_act = np.append(limit_act, self.action_max)

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
        self.previous_power = {}
        self.previous_voltage = {}
        self.previous_action = {}
        self.previous_angles = {}
        self.mean_basepressure = None
        self.previous_abspdiff = None
        
    def Obs_FS(self, N_stack, N_skip=1, probe_nums = 7, action_nums = 1, angle_nums = 2):
        
        # Function implementing framestack with frame skipping.
        # This helps extend the horizon into the past with a relatively high frequency of sampling frames.
        observations = np.array([])
        buffer_required = N_stack * N_skip;
        if buffer_required > self.size_history:
            raise Exception("Buffer size too small for FrameStack.")
        
        past_pressure = np.zeros(probe_nums * N_stack)
        past_action = np.zeros(action_nums * N_stack)
        past_angle = np.zeros(angle_nums * N_stack)
        past_pdiff = np.zeros(1 * N_stack)
            
        for i in range(N_stack) :
            
            if N_skip ==0:
                assert 'invalid value for N_skip'
            index_value = -1 if i == 0 else -i * N_skip -1 #if i = 0, take the current obseravtion
            
            
            
            # Obtain past pressure measurements
            past_pressure[i*probe_nums : i*probe_nums + probe_nums] = [self.history_parameters["sensor_{}".format(num_sensors)].get_value(index_value)
                                   for num_sensors in range(probe_nums)]


                                   
            
            past_action[i*action_nums : i*action_nums + action_nums] = [self.history_parameters["action_{}".format(num_actions)].get_value(index_value)
                                    for num_actions in range(action_nums)]
                                    

            

        if self.include_anlge == True:
            observations = np.concatenate((past_pressure,  past_action, past_angle))
        else:
            observations = np.concatenate((past_pressure,  past_action))
        

        return observations

        
    def write_history_parameters(self):
        '''
        Add data of last step to history buffers
        :return:
        '''

        self.history_parameters["drag"].extend(np.array(self.previous_ForceX))
        self.history_parameters["lift"].extend(np.array(self.previous_ForceZ))

        
        self.history_parameters["mean_basepressure"].extend(np.array(self.mean_basepressure))
        self.history_parameters["pdiff_abs"].extend(np.array(self.meanabs_pdiff))
        self.history_parameters["encoder1"] .extend(np.array(self.encoder1))
        
        #self.history_parameters["action_1"].extend(np.array([self.previous_action]))
        
        
    def _receive_payload(self):

        if self.include_anlge == True:
            #payload = self.env_server.receive_payload(">dddddddddddddddddddddd")
            num_d = 69  # Specify the number of 'd' characters you need
            format_string = ">" + "d" * num_d
            payload = self.env_server.receive_payload(format_string)


        formatted_payload = self.format_payload(payload)
        #print(formatted_payload)
        
 
        self.previous_ForceX = formatted_payload[0]
        self.previous_ForceY = formatted_payload[1] 
        self.previous_ForceZ = formatted_payload[2]
        #self.encoder1 = formatted_payload[-1]
        

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
            actions =  np.array([action[0], action[0]])
            #actions =  np.array([20, 20])
            
        
        
        else:
            assert 'flap behavior defiend incorrectly'
            

        for num_flaps in range(2):        
            self.previous_action[num_flaps] = actions[num_flaps]
        #initial = time.time()

        self.env_server.send_payload(
            payload = actions,
            #sending_mask = ">dddd"
            sending_mask = ">dd"
        )

        received_data = self._receive_payload()
        
        
        
        current_pressure = np.zeros(64,) 
        current_angles = np.zeros(2,)
        current_pdiff = np.zeros(28,)
        current_base_pressure = np.zeros(36,)
        
            
        left_indices = [3,4,5,9,10,11,15,16,17,21,22,23,27,28,29,33,34,35]  #left indices of the base
        right_indices = [6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38] #right indices of the base
        
        current_pdiff[0] = np.mean(received_data[left_indices]) - np.mean(received_data[right_indices])
        current_pdiff[1:28] = received_data[39:66]
        current_angles = received_data[67:69]
        current_pressure = received_data[3:67]
        current_base_pressure = received_data[3:39]
        


        self.mean_basepressure = np.mean(current_base_pressure)
        self.meanabs_pdiff = current_pdiff[0]
        self.encoder1 = received_data[68]
        self.episodic_bp += self.mean_basepressure
        self.episodic_pdiff += self.meanabs_pdiff

        

        
        current_ForceX = received_data[0]
        current_ForceY = received_data[1]
        current_ForceZ = received_data[2]
        
        # Write latest data to buffer
        self.write_history_parameters()
        
        self.avg_window = min(self.n_step, 50)
        


        if self.reward_function == 'force_reward':
            self.reward =  abs(self.avg_baseline_drag[0]) - abs(current_ForceX - self.reward_compensation) - 2*abs(current_ForceY)#2*abs(avg_ForceY)
            
        elif self.reward_function == 'avg_force_reward':
            avg_ForceY = np.mean(self.history_parameters["side"].get()[-self.avg_window:])
            avg_ForceX = np.mean(self.history_parameters["drag"].get()[-self.avg_window:])
            #if self.last_epi_trigger == False:
            self.reward = abs(self.avg_baseline_drag[0]) - abs(avg_ForceX - self.reward_compensation) - 0.6*abs(avg_ForceY)#2*abs(avg_ForceY)
            #else:
            #    self.reward = abs(self.avg_baseline_drag[0]) - abs(avg_ForceX - self.reward_compensation) - 0.3*abs(avg_ForceY)
                 
        elif self.reward_function == 'base_pressure_reward':
            
            self.reward = np.mean(self.history_parameters["mean_basepressure"].get()[-self.avg_window:]) - self.reward_compensation 
            - 1 * np.mean(self.history_parameters["pdiff_abs"].get()[-self.avg_window:]) 
            
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
            self.reward = np.mean(self.history_parameters["abs_shift_pd"].get()[-self.avg_window:]) - 0.5 * instant_change
            
        
        
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
            observations = np.concatenate((observations, np.array([current_pdiff[0]])))
                
        if self.include_anlge == True:
            
            observations = np.concatenate((observations,current_angles))
        
        if self.include_action == True:
            
            observations = np.concatenate((observations, np.array([action[-1]])))
        
            
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
        self.history_parameters["mean_basepressure"] = RingBuffer(self.size_history)
        self.history_parameters["pdiff_abs"] = RingBuffer(self.size_history)
        self.history_parameters["encoder1"] = RingBuffer(self.total_steps)
        
        
        self.history_parameters["baseline_basepressure"] = RingBuffer(self.baseline_duras)
        self.history_parameters["baseline_pdiff"] = RingBuffer(self.baseline_duras)

        


        self._logger.warning("ENV RESET")
        
        
        #self.env_server.send_payload([0, 0, 0, 0], sending_mask=">dddd") # 
        self.env_server.send_payload([0, 0], sending_mask=">dd") # 
        #print('#indicator')

        time.sleep(6)
        
        if self.n_episodes % self.interval_record_baseline == 0:

            self._logger.info(f"Obtain Baseline")
            time.sleep(0.25)
            self.baseline_bp = 0
            print('#entering baseline recording indicator')
            
            for i in range(self.baseline_duras):
                baseline_data = self._receive_payload()
                
                time.sleep(0.025)
                #self.history_parameters["baseline_drag"].extend(baseline_data[7])
                #self.history_parameters["baseline_lift"].extend(baseline_data[9])
                #self.history_parameters["baseline_side"].extend(baseline_data[8])
                
                left_indices = [3,4,5,9,10,11,15,16,17,21,22,23,27,28,29,33,34,35]  #left indices of the base
                right_indices = [6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38] #right indices of the base
                
                baseline_base_pressure = np.zeros(36,)
                baseline_base_pressure = baseline_data[3:39]
                self.baseline_pdiff = np.mean(baseline_data[left_indices]) - np.mean(baseline_data[right_indices])
                    
                self.mean_baseline_basepressure = np.mean(baseline_base_pressure)
                self.baseline_bp += self.mean_baseline_basepressure
                
                self.history_parameters["baseline_basepressure"].extend(self.mean_baseline_basepressure)
                self.history_parameters["baseline_pdiff"].extend(self.baseline_pdiff)
                #self.history_parameters["pdiff_abs"].extend(self.baseline_pdiff)
                
                
            self.avg_baseline_pressure = np.append(self.avg_baseline_pressure, np.mean(self.history_parameters["baseline_basepressure"].get()))
            self.avg_baseline_pdiff = np.append(self.avg_baseline_pdiff, np.mean(self.history_parameters["baseline_pdiff"].get()))
            
            
                           
            if self.avg_baseline_pressure.size > 1:
                
                self._logger.info(f"Compensate rewards")
                self.reward_compensation = self.avg_baseline_pressure[-1] - self.avg_baseline_pressure[0]
                print("Pressure drifting", self.reward_compensation)
                
                
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
                                  + "\t"+ 'Episodic Pdiff' + "\t"+ 'Average Action Frequency')
                
                WriteReward.write("\n")
                
            if self.reward_shaping_trigger == True:
                
                WriteReward.write(str(self.n_episodes)+"\t" + str(self.total_reward) + "\t" + str(self.episodic_bp) 
                                  + "\t"+ str(self.episodic_pdiff) + "\t"+ str(self.total_actionPF/(self.total_steps-self.periodic_window)))
                
            else:
                
                WriteReward.write(str(self.n_episodes)+"\t" + str(self.total_reward) + "\t" + str(self.episodic_bp) 
                                  + "\t"+ str(self.episodic_pdiff) + "\t"+ str(self.total_actionPF/(self.total_steps-self.periodic_window)))
                
            WriteReward.write("\n")
            
            

        self.total_reward = 0
        self.episodic_bp = 0
        self.episodic_pdiff = 0
        self.total_actionPF = 0

    

        #payload = self.env_server.receive_payload(">dddddddddddddddddd")
        #received_data = self.format_payload(payload)
        
        
        received_data = self._receive_payload()
        
        

        self._logger.warning("ENV RESET DONE")


        current_actions = np.zeros(4,)
        current_pressure = np.zeros(64,) 
        current_angles = np.zeros(2,)
        current_pdiff = np.zeros(28,)
        current_base_pressure = np.zeros(36,)
        
            
        left_indices = [3,4,5,9,10,11,15,16,17,21,22,23,27,28,29,33,34,35]  #left indices of the base
        right_indices = [6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38] #right indices of the base
        
        current_pdiff[0] = np.mean(received_data[left_indices]) - np.mean(received_data[right_indices])
        current_pdiff[1:28] = received_data[39:66]
        current_angles = received_data[67:69]
        current_pressure = received_data[3:67]
        current_base_pressure = received_data[3:39]
            

        current_ForceX = received_data[0]
        current_ForceY = received_data[1]
        current_ForceZ = received_data[2]
        
        self.mean_basepressure = np.mean(current_base_pressure)
        self.meanabs_pdiff = np.absolute(current_pdiff[0])
        self.encoder1 = received_data[68]
        
        # Write latest data to buffer
        self.write_history_parameters()
        
            
        if self.pdiff_obs ==True:
            observations = current_pdiff
        else:
            observations = current_pressure
            observations = np.concatenate((observations, np.array([current_pdiff[0]])))
                
        if self.include_anlge == True:
            
            observations = np.concatenate((observations,current_angles))
            
        if self.include_action == True:
            observations = np.concatenate((observations, np.zeros(self.action_shape,)))
            
        
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
    

