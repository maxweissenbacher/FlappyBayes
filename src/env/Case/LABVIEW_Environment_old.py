import logging
import numpy as np

from gym import spaces
from DrlPlatform import PlatformEnvironment
import time



start = time.time()

class RingBuffer():

    def __init__(self,length):
        self.data = np.zeros(length, dtype='d')
        self.index = 0

    def extend(self,x):
        x_indices = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_indices] = x

    def get(self):
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]

class AhmedBody_Force(PlatformEnvironment):
    """Custom class that represents the Ahmed body environment:

    Observations:
        ????
    Actions:
        ????

    """
    def __init__(self):


        #Parameters of the flow

        rho = 1.225
        vel = 30
        ref_Area = 200 #to be checked

        super(AhmedBody_Force, self).__init__()

        self.size_history = 500
        self.start_class()
        self._logger = logging.getLogger(__name__)
        self.lift_max = np.finfo(np.float32).max
        self.drag_max = np.finfo(np.float32).max
        self.action1_max = 5
        self.action2_max = 5
        limit_obs = np.array(
            [
                self.lift_max,
                self.drag_max,
            ],
            dtype=np.float32,
        )
        limit_act = np.array(
            [
                self.action1_max,
                self.action2_max,
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(-limit_obs, limit_obs, dtype=np.float32)  # type: ignore
        self.action_space = spaces.Box(-limit_act, limit_act, dtype=np.float32)
        self.total_reward = 0.0
        self.done = False

        # Historical data
        self.previous_lift = None
        self.previous_drag = None
        self.previous_action_1 = None
        self.previous_action_2 = None

    def start_class(self):
        self.history_parameters = {}
        self.history_parameters["drag"] = RingBuffer(self.size_history)
        self.history_parameters["lift"] = RingBuffer(self.size_history)

    def write_history_parameters(self):
        self.history_parameters["drag"].extend(np.array(self.previous_drag))
        self.history_parameters["lift"].extend(np.array(self.previous_lift))

    def _receive_payload(self):

        # Returns current_lift, current_drag

        self.lift_threshold = 10000
        self.drag_threshold = 10000
        payload = self.env_server.receive_payload(">dd")
        formatted_payload = self.format_payload(payload)
        current_lift, current_drag = \
            formatted_payload[0], formatted_payload[1]
        if not (self.previous_lift and self.previous_drag):
            self.previous_lift = current_lift
            self.previous_drag = current_drag
        else:
            # Noise detection
            while True:
                if abs(self.previous_lift - current_lift) > self.lift_threshold or \
                        abs(self.previous_drag - current_drag) > self.drag_threshold:
                    self._logger.warning("Detected noise")
                    payload = self.env_server.receive_payload(">dd")
                    formatted_payload = self.format_payload(payload)
                    current_lift, current_drag = \
                        formatted_payload[0], formatted_payload[1]
                    continue
                else:
                    break
        self.previous_lift = current_lift
        self.previous_drag = current_drag

        self.write_history_parameters()

        return current_lift, current_drag

    def step(self, action):
        # Negative reward for a step
        info = {}
        # Sending action through env server

        if action is None:
            no_flaps=2
            action = np.zeros((no_flaps,))
        self.previous_action_1 = action[0]
        self.previous_action_2 = action[1]
        initial = time.time()

        self.env_server.send_payload(
            payload = action,
            sending_mask = ">dd"
        )
        sent = time.time()
        print("Sent Elapsed: ", (sent - start))


        current_lift, current_drag = self._receive_payload()
        # Compute episodic reward (as a function of CL and CD)
        receive = time.time()
        print("Received Elapsed: ", (receive - start))
        print("Difference: ", (receive - sent))

        previous_lift = self.history_parameters["lift"].get()[-1]
        previous_drag = self.history_parameters["drag"].get()[-1]

        print("Previous Lift:", previous_lift)
        print("Previous Drag:", previous_drag)


        self.reward = -0.2 * self._lift_reward(current_lift) - self._drag_reward(current_drag)
        self.total_reward += self.reward
        formatted_payload = np.array([current_lift, current_drag],
                                     dtype=np.float32)

        # Return the result
        self._logger.info(f"Reward: {self.reward}")
        #self._logger.info(f"Accumulated Reward: {self.total_reward}")
        return formatted_payload, self.reward, self.done, info

    def _lift_reward(self, lift: float):

        rho = 1.225
        vel = 30
        ref_Area = 0.001 #to be checked
        return lift / (0.5 * rho * (vel**2) * ref_Area)

    def _drag_reward(self, drag: float):

        rho = 1.225
        vel = 30
        ref_Area = 0.001 #to be checked
        return drag / (0.5 * rho * (vel**2) * ref_Area)

    def reset(self) -> np.ndarray:
        self.counter = 0
        self.total_reward = 0
        self.done = False
        self.previous_lift = None
        self.previous_drag = None
        self.start_class()
        self._logger.warning("ENV RESET")
        if not self.previous_action_1 and not self.previous_action_2:
            self.previous_action_1 = 0
            self.previous_action_2 = 0
        self.env_server.send_payload([self.previous_action_1, self.previous_action_2],
                                     sending_mask=">dd")
        self.previous_action_1 = None
        self.previous_action_2 = None
        #time.sleep(0.031)
        payload = self.env_server.receive_payload(">dd")
        formatted_payload = self.format_payload(payload)
        while not self._is_env_reset(formatted_payload):
            self.env_server.send_payload([0, 0],
                                         sending_mask=">dd")
            #time.sleep(0.031)
            payload = self.env_server.receive_payload(">dd")
            formatted_payload = self.format_payload(payload)
        self._logger.warning("ENV RESET DONE")
        current_lift, current_drag = formatted_payload[0], formatted_payload[1]
        formatted_payload = np.array([current_lift, current_drag],
                                     dtype=np.float32)
        return formatted_payload

    def format_payload(self, payload) -> np.ndarray:
        """
        Formatting payload which is going to be sent to training agent
        """
        formatted_payload = np.array(payload, dtype=np.float32)

        current_lift, current_drag = formatted_payload[0], formatted_payload[1]

        return np.array([current_lift, current_drag], dtype=np.float32)

    def _is_env_reset(self, formatted_payload) -> bool:
        """
        Check if env has been reset
        """
        current_lift, current_drag = formatted_payload[0], formatted_payload[1]
        counter = 0
        is_reset = True
        while counter <= 5:

            # How to find a proper condition to check if the environment has been reset?

            counter += 1
        return is_reset

class AhmedBody_Pressure(PlatformEnvironment):
    """Custom class that represents the Ahmed body environment:

    Observations:
        ????
    Actions:
        ????

    """
    def __init__(self):


        #Parameters of the flow

        rho = 1.225
        vel = 30
        ref_Area = 200 #to be checked

        super(AhmedBody_Pressure, self).__init__()
        self._logger = logging.getLogger(__name__)
        self.pressure_max = np.finfo(np.float32).max
        self.action1_max = 5
        self.action2_max = 5

        limit_obs = np.array(
            [
                self.pressure_max,
                self.pressure_max,
                self.pressure_max,
                self.pressure_max,
                self.pressure_max,
                self.pressure_max,
                self.pressure_max,

            ],
            dtype=np.float32,
        )
        limit_act = np.array(
            [
                self.action1_max,
                self.action2_max,
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(-limit_obs, limit_obs, dtype=np.float32)  # type: ignore
        self.action_space = spaces.Box(-limit_act, limit_act, dtype=np.float32)
        self.total_reward = 0.0
        self.done = False

        # Historical data
        self.previous_pressure_1 = None
        self.previous_pressure_2 = None
        self.previous_pressure_3 = None
        self.previous_pressure_4 = None
        self.previous_pressure_6 = None
        self.previous_pressure_7 = None
        self.previous_pressure_8 = None


        self.previous_action_1 = None
        self.previous_action_2 = None

    def _receive_payload(self):

        # Returns history_lift, history_drag, current_lift, current_drag


        payload = self.env_server.receive_payload(">ddddddd")
        formatted_payload = self.format_payload(payload)
        current_pressure_1, current_pressure_2, current_pressure_3,\
            current_pressure_4, current_pressure_6, current_pressure_7, current_pressure_8 = \
            formatted_payload[0], formatted_payload[1], formatted_payload[2], formatted_payload[3], formatted_payload[4], \
                formatted_payload[5], formatted_payload[6]
        if not (self.previous_pressure_1):
            self.previous_pressure_1 = current_pressure_1
            self.previous_pressure_2 = current_pressure_2
            self.previous_pressure_3 = current_pressure_3
            self.previous_pressure_4 = current_pressure_4
            self.previous_pressure_6 = current_pressure_6
            self.previous_pressure_7 = current_pressure_7
            self.previous_pressure_8 = current_pressure_8


        self.previous_pressure_1 = current_pressure_1
        self.previous_pressure_2 = current_pressure_2
        self.previous_pressure_3 = current_pressure_3
        self.previous_pressure_4 = current_pressure_4
        self.previous_pressure_6 = current_pressure_6
        self.previous_pressure_7 = current_pressure_7
        self.previous_pressure_8 = current_pressure_8

        return current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4,\
            current_pressure_6, current_pressure_7, current_pressure_8
    def step(self, action):
        # Negative reward for a step
        info = {}
        # Sending action through env server

        if action is None:
            no_flaps=2
            action = np.zeros((no_flaps,))
        self.previous_action_1 = action[0]
        self.previous_action_2 = action[1]
        initial = time.time()

        self.env_server.send_payload(
            payload = action,
            sending_mask = ">dd"
        )
        sent = time.time()
        print("Sent Elapsed: ", (sent - start))

        current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4, \
            current_pressure_6, current_pressure_7, current_pressure_8 = self._receive_payload()
        # Compute episodic reward (as a function of CL and CD)
        receive = time.time()
        print("Received Elapsed: ", (receive - start))
        print("Time to do the entire Labview loop: ", (receive - sent))
        self.reward = (current_pressure_1 + current_pressure_2 + current_pressure_3 + current_pressure_4 + \
            current_pressure_6 + current_pressure_7 + current_pressure_8) * 200/8
        self.total_reward += self.reward
        formatted_payload = np.array([ current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4,\
            current_pressure_6, current_pressure_7, current_pressure_8],
                                     dtype=np.float32)

        # Return the result
        #self._logger.info(f"Reward: {self.reward}")
        return formatted_payload, self.reward, self.done, info

    def _lift_reward(self, lift: float):

        rho = 1.225
        vel = 30
        ref_Area = 0.001 #to be checked
        return lift / (0.5 * rho * (vel**2) * ref_Area)

    def _drag_reward(self, drag: float):

        rho = 1.225
        vel = 30
        ref_Area = 0.001 #to be checked
        return drag / (0.5 * rho * (vel**2) * ref_Area)

    def reset(self) -> np.ndarray:
        self.counter = 0
        self.total_reward = 0
        self.done = False
        self.previous_pressure_1 = None
        self.previous_pressure_2 = None
        self.previous_pressure_3 = None
        self.previous_pressure_4 = None
        self.previous_pressure_6 = None
        self.previous_pressure_7 = None
        self.previous_pressure_8 = None
        self._logger.warning("ENV RESET")
        if not self.previous_action_1 and not self.previous_action_2:
            self.previous_action_1 = 0
            self.previous_action_2 = 0
        self.env_server.send_payload([self.previous_action_1, self.previous_action_2],
                                     sending_mask=">dd")
        self.previous_action_1 = None
        self.previous_action_2 = None
        #time.sleep(0.031)
        payload = self.env_server.receive_payload(">ddddddd")
        formatted_payload = self.format_payload(payload)
        #while not self._is_env_reset(formatted_payload):
        #    self.env_server.send_payload([0, 0],
        #                                 sending_mask=">dd")
            #time.sleep(0.031)
        #    payload = self.env_server.receive_payload(">dddddddddd")
        #    formatted_payload = self.format_payload(payload)
        self._logger.warning("ENV RESET DONE")
        current_pressure_1, current_pressure_2, current_pressure_3, \
            current_pressure_4, current_pressure_6, current_pressure_7, current_pressure_8 = \
                formatted_payload[0], formatted_payload[1], formatted_payload[2], formatted_payload[3], \
                formatted_payload[4], formatted_payload[5], formatted_payload[6]
        formatted_payload = np.array([current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4,\
            current_pressure_6, current_pressure_7, current_pressure_8],
                                     dtype=np.float32)

        return formatted_payload

    def format_payload(self, payload) -> np.ndarray:
        """
        Formatting payload which is going to be sent to training agent
        """
        formatted_payload = np.array(payload, dtype=np.float32)

        current_pressure_1, current_pressure_2, current_pressure_3, \
            current_pressure_4, current_pressure_6, current_pressure_7, current_pressure_8 = \
            formatted_payload[0], formatted_payload[1], formatted_payload[2], formatted_payload[3], \
            formatted_payload[4], formatted_payload[5], formatted_payload[6]
        return np.array([current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4,\
            current_pressure_6, current_pressure_7, current_pressure_8], dtype=np.float32)

    #def _is_env_reset(self, formatted_payload) -> bool:
     #   """
     #   Check if env has been reset
     #   """
     #   current_lift, current_drag = formatted_payload[0], formatted_payload[1]
     #   counter = 0
     #   is_reset = True
     #   while counter <= 5:

            # How to find a proper condition to check if the environment has been reset?

     #       counter += 1
     #   return is_reset

class AhmedBody_AllObservations(PlatformEnvironment):

    def __init__(self):



        super(AhmedBody_AllObservations, self).__init__()
        self._logger = logging.getLogger(__name__)
        self.pressure_max = np.finfo(np.float32).max
        self.ForceX_max = np.finfo(np.float32).max
        self.ForceY_max = np.finfo(np.float32).max
        self.ForceZ_max = np.finfo(np.float32).max
        self.power_max = np.finfo(np.float32).max
        self.voltage_max = np.finfo(np.float32).max
        self.action1_max = 5
        self.action2_max = 5
        self.action3_max = 5
        self.action4_max = 5

        limit_obs = np.array(
            [
                self.pressure_max,
                self.pressure_max,
                self.pressure_max,
                self.pressure_max,
                self.pressure_max,
                self.pressure_max,
                self.pressure_max,
                #self.ForceX_max,
                #self.ForceY_max,
                #self.ForceZ_max,
                self.voltage_max,
                self.voltage_max,
                self.voltage_max,
                self.voltage_max

            ],
            dtype=np.float32,
        )
        limit_act = np.array(
            [
                self.action1_max,
                self.action2_max,
                self.action3_max,
                self.action4_max,
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(-limit_obs, limit_obs, dtype=np.float32)  # type: ignore
        self.action_space = spaces.Box(-limit_act, limit_act, dtype=np.float32)
        self.total_reward = 0.0
        self.total_drag = 0.0
        self.done = False
        self.n_episode = -1
        self.prev_time = 0
        # Historical data
        self.previous_pressure_1 = None
        self.previous_pressure_2 = None
        self.previous_pressure_3 = None
        self.previous_pressure_4 = None
        self.previous_pressure_6 = None
        self.previous_pressure_7 = None
        self.previous_pressure_8 = None
        self.previous_ForceX = None
        self.previous_ForceY= None
        self.previous_ForceZ = None
        self.previous_power_1 = None
        self.previous_power_2 = None
        self.previous_power_3 = None
        self.previous_power_4 = None
        self.previous_voltage_1 = None
        self.previous_voltage_2 = None
        self.previous_voltage_3 = None
        self.previous_voltage_4 = None


        self.previous_action_1 = None
        self.previous_action_2 = None
        self.previous_action_3 = None
        self.previous_action_4 = None

    def _receive_payload(self):

        # Returns history_lift, history_drag, current_lift, current_drag


        payload = self.env_server.receive_payload(">dddddddddddddddddd")
        formatted_payload = self.format_payload(payload)
        current_pressure_1, current_pressure_2, current_pressure_3,\
            current_pressure_4, current_pressure_6, current_pressure_7, current_pressure_8,\
            current_ForceX, current_ForceY, current_ForceZ, current_power_1, current_power_2,\
            current_power_3, current_power_4, current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4= \
            formatted_payload[0], formatted_payload[1], formatted_payload[2], formatted_payload[3], formatted_payload[4], \
                formatted_payload[5], formatted_payload[6], formatted_payload[7], formatted_payload[8], formatted_payload[9], \
                formatted_payload[10], formatted_payload[11], formatted_payload[12], formatted_payload[13], \
                formatted_payload[14], formatted_payload[15], formatted_payload[16], formatted_payload[17]


        self.previous_pressure_1 = current_pressure_1
        self.previous_pressure_2 = current_pressure_2
        self.previous_pressure_3 = current_pressure_3
        self.previous_pressure_4 = current_pressure_4
        self.previous_pressure_6 = current_pressure_6
        self.previous_pressure_7 = current_pressure_7
        self.previous_pressure_8 = current_pressure_8
        self.previous_ForceX = current_ForceX
        self.previous_ForceY = current_ForceY
        self.previous_ForceZ = current_ForceZ
        self.previous_power_1 = current_power_1
        self.previous_power_2 = current_power_2
        self.previous_power_3 = current_power_3
        self.previous_power_4 = current_power_4
        self.previous_voltage_1 = current_voltage_1
        self.previous_voltage_2 = current_voltage_2
        self.previous_voltage_3 = current_voltage_3
        self.previous_voltage_4 = current_voltage_4

        return current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4,\
            current_pressure_6, current_pressure_7, current_pressure_8, current_ForceX, current_ForceY, \
            current_ForceZ, current_power_1, current_power_2, current_power_3, current_power_4, \
            current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4
    def step(self, action):
        # Negative reward for a step
        info = {}
        Area = 0.216 * 0.160
        Velocity = 15
        baseline_ForceX = -1.499
        # Sending action through env server3

        if action is None:
            no_flaps=4
            action = np.zeros((no_flaps,))
        self.previous_action_1 = action[0]
        self.previous_action_2 = action[1]
        self.previous_action_3 = action[2]
        self.previous_action_4 = action[3]
        #initial = time.time()

        self.env_server.send_payload(
            payload = action,
            sending_mask = ">dddd"
        )
        #sent = time.time()
        #print("Sent Elapsed: ", (sent - start))
        #loop_time = sent - self.prev_time
        #print("Loop time: ", loop_time)
        #self.prev_time = sent

        current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4, \
            current_pressure_6, current_pressure_7, current_pressure_8, current_ForceX, current_ForceY, \
            current_ForceZ, current_power_1, current_power_2, current_power_3, current_power_4, \
            current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4 = self._receive_payload()

        #receive = time.time()
        #print("Received Elapsed: ", (receive - start))


        #loop_time = receive - self.prev_time
        #print("Loop time: ", loop_time)
        #self.prev_time = receive

        #receive = time.time()
        #print("Received Elapsed: ", (receive - start))
        #print("Time to do the entire Labview loop: ", (receive - sent))
        # Compute episodic reward (as a function of CL and CD)
        #self.reward = (current_pressure_1 + current_pressure_2 + current_pressure_3 + current_pressure_4 + \
         #current_pressure_6 + current_pressure_7 + current_pressure_8) * Area / 7
        self.reward = (abs(baseline_ForceX) - abs(current_ForceX)) * Velocity  #- (current_power_1 + current_power_2 + current_power_3 + current_power_4) + (abs(baseline_ForceX) - abs(current_ForceX)) * Velocity
        #self.reward = self.reward = -0.2 * self._lift_reward(current_ForceZ) - self._drag_reward(current_ForceX)
        self.total_reward += self.reward
        self.total_drag += abs(current_ForceX)
        observations = np.array([ current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4,\
            current_pressure_6, current_pressure_7, current_pressure_8, \
            current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4],
                                     dtype=np.float32)

        self._logger.info(f"Reward: {self.reward}")

        return observations, self.reward, self.done, info


    def reset(self) -> np.ndarray:
        self.counter = 0

        self.done = False
        self.previous_pressure_1 = None
        self.previous_pressure_2 = None
        self.previous_pressure_3 = None
        self.previous_pressure_4 = None
        self.previous_pressure_6 = None
        self.previous_pressure_7 = None
        self.previous_pressure_8 = None
        self.previous_ForceX = None
        self.previous_ForceY= None
        self.previous_ForceZ = None
        self.previous_power_1 = None
        self.previous_power_2 = None
        self.previous_power_3 = None
        self.previous_power_4 = None
        self.previous_voltage_1 = None
        self.previous_voltage_2 = None
        self.previous_voltage_3 = None
        self.previous_voltage_4 = None

        self._logger.warning("ENV RESET")
        #if not self.previous_action_1 and not self.previous_action_2 and not self.previous_action_3 and not self.previous_action_4 :
        self.previous_action_1 = 0
        self.previous_action_2 = 0
        self.previous_action_3 = 0
        self.previous_action_4 = 0
        self.env_server.send_payload([self.previous_action_1, self.previous_action_2, self.previous_action_3, self.previous_action_4],
                                     sending_mask=">dddd")
        #self.previous_action_1 = 0
        #self.previous_action_2 = 0
        #self.previous_action_3 = 0
        #self.previous_action_4 = 0

        self.n_episode += 1
        self._logger.info(f"Episode Number: {self.n_episode}")

        with open(f"reward.txt", "a") as WriteReward:
            WriteReward.write(str(self.n_episode)+"\t"+str(self.total_reward)+"\t"+str(self.total_drag))
            WriteReward.write("\n")

        self.total_reward = 0
        self.total_drag = 0

        payload = self.env_server.receive_payload(">dddddddddddddddddd")
        formatted_payload = self.format_payload(payload)
        time.sleep(3)

        self._logger.warning("ENV RESET DONE")

        current_pressure_1, current_pressure_2, current_pressure_3, \
            current_pressure_4, current_pressure_6, current_pressure_7, current_pressure_8, \
            current_ForceX, current_ForceY, current_ForceZ, current_power_1, current_power_2, \
            current_power_3, current_power_4, current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4 = \
            formatted_payload[0], formatted_payload[1], formatted_payload[2], formatted_payload[3], formatted_payload[4], \
                formatted_payload[5], formatted_payload[6], formatted_payload[7], formatted_payload[8], formatted_payload[9], \
                formatted_payload[10], formatted_payload[11], formatted_payload[12], formatted_payload[13], \
                formatted_payload[14], formatted_payload[15], formatted_payload[16], formatted_payload[17]

        observations = np.array([current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4,\
            current_pressure_6, current_pressure_7, current_pressure_8, \
            current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4],
                                     dtype=np.float32)
        return observations

    def format_payload(self, payload) -> np.ndarray:
        """
        Formatting payload which is going to be sent to training agent
        """
        formatted_payload = np.array(payload, dtype=np.float32)

        current_pressure_1, current_pressure_2, current_pressure_3, \
            current_pressure_4, current_pressure_6, current_pressure_7, current_pressure_8, \
            current_ForceX, current_ForceY, current_ForceZ, current_power_1, current_power_2, \
            current_power_3, current_power_4, current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4 = \
            formatted_payload[0], formatted_payload[1], formatted_payload[2], formatted_payload[3], formatted_payload[4], \
                formatted_payload[5], formatted_payload[6], formatted_payload[7], formatted_payload[8], formatted_payload[9], \
                formatted_payload[10], formatted_payload[11], formatted_payload[12], formatted_payload[13], \
                formatted_payload[14], formatted_payload[15], formatted_payload[16], formatted_payload[17]

        return np.array([current_pressure_1, current_pressure_2, current_pressure_3, current_pressure_4,\
            current_pressure_6, current_pressure_7, current_pressure_8, current_ForceX, current_ForceY, current_ForceZ, \
                 current_power_1, current_power_2, current_power_3, current_power_4, \
                         current_voltage_1, current_voltage_2, current_voltage_3, current_voltage_4], dtype=np.float32)

