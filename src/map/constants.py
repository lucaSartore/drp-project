import math
import numpy as np

MAP_X_LOWER_BOUND: float = -10.0
MAP_X_UPPER_BOUND: float =  10.0
MAP_Y_LOWER_BOUND: float = -10.0
MAP_Y_UPPER_BOUND: float =  10.0


MAP_AREA = (MAP_X_UPPER_BOUND-MAP_X_LOWER_BOUND) * (MAP_Y_UPPER_BOUND-MAP_Y_LOWER_BOUND)

RUNNER_VELOCITY: float = 0.05
CHASER_VELOCITY: float = 0.05

BOUNCING_RANDOMNESS: float = math.pi/32 # rad

MEASUREMENT_STD = 0.5
MEASUREMENT_COVARIANCE = np.array(
    [[MEASUREMENT_STD ** 2, 0],
     [0, MEASUREMENT_STD ** 2]],
    dtype=np.float32
)

PARTICLE_UPDATE_COVARIANCE = 3 * np.array(
    [[RUNNER_VELOCITY ** 2, 0],
     [0, RUNNER_VELOCITY ** 2]],
    dtype=np.float32
)
