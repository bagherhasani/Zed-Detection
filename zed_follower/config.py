import numpy as np


class FollowerConfig:
    def __init__(self):

        # how far robot be from perosn (metres)
        self.target_distance = 1.0
        self.min_distance = 0.6  # stop if closer than this 

        # max speeds 
        self.max_lin_speed = 0.4   # m/s forward
        self.max_ang_speed = 0.8   # rad/s turning

        #  gains 
        self.k_lin = 0.7
        self.k_ang = 1.0

        # search behaviour when trartget lost
        self.search_enabled = True
        self.search_turn_speed = 0.5
        self.search_timeout_sec = 10.0

        # how long to wait before fully forgetting the locked person
        self.lock_lost_timeout_sec = 30.0

        # shirt colour in HSV — blue shirt (H: 108-122, S: 100-255, V: 40-255)
        self.target_hsv_lower = np.array([108, 100, 40])
        self.target_hsv_upper = np.array([122, 255, 255])

       
