CHEBYSHEV_ORDER_X = 10
CHEBYSHEV_ORDER_Y = 10
NUMBER_OF_APPROXIMATION_COEFFICIENTS = (CHEBYSHEV_ORDER_X+1) * (CHEBYSHEV_ORDER_Y+1)
NUMBER_OF_PARTICLES = 2000
NUMBER_OF_PARTICLES_TO_RESAMPLE_RANDOMLY = 400
WEIGHTS_TO_ASSIGN_TO_RANDOMLY_SAMPLED_PARTICLES = 0.01
# the number of consensus iteration should be higher
# if the connection structure isn't a "mesh" however
# if it is a mesh there is no advantage in increasing the number
# as they will always converge in one iteration
CONSENSUS_ITERATIONS = 1
DEBUG = False

NUM_EXTRA_PARTICLES_PER_SIDE = 25
EXTRA_PARTICLE_BORDER_DISTANCE = 0.1

MIN_DISTANCE_FOR_VALID_CATCH = 0.15
