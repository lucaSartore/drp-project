import numpy as np
from map.constants import MAP_AREA, MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND, MEASUREMENT_COVARIANCE
from chasers_logic.constants import NUMBER_OF_PARTICLES, CHEBYSHEV_ORDER_X, CHEBYSHEV_ORDER_Y
from map.data_type import Point
from map.map import Settings
from scipy.stats import multivariate_normal
from numpy.polynomial.chebyshev import chebvander2d


class ParticleFilterManager:
    def __init__(
        self,
        number_of_agents: int,
        agent_id: int,
        settings: Settings
    ) -> None:
        self.number_of_agents = number_of_agents
        self.agent_id = agent_id
        self.particles = np.zeros(shape=(NUMBER_OF_PARTICLES, 2), dtype=np.float32)
        """
        particles in the particle filters
        shape: [NUMBER_OF_PARTICLES, 2] (where 2 is x,y)
        """

        rand = lambda lb, ub: np.random.random(NUMBER_OF_PARTICLES) * (ub - lb) + lb
        self.particles[:,0] = rand(MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND)
        self.particles[:,1] = rand(MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND)

        self.weights = np.ones(shape=(NUMBER_OF_PARTICLES), dtype=np.float32) * 1/NUMBER_OF_PARTICLES
        """
        weights associated with each particle
        shape: [NUMBER_OF_PARTICLES]
        """

        self.settings = settings


    def _get_particle(self, index: int) -> Point:
        return Point(
            self.particles[index,0],
            self.particles[index,1]
        )

    def _probability_of_measure(self, measure: Point | None, position: Point, particle_index: int) -> float:
        particle = self._get_particle(particle_index)

        # the measure is null (no detection)
        if measure == None:
            is_in_radius = (position-particle).abs() >= self.settings.chaser_detection_radius 
            probability_real_runner = 0 if not is_in_radius else self.settings.runner_false_negative_probability


            # approximation of the area covered by the radar 
            # (is approximated as we are not removing the part outside the map
            # if the robot is close to the border)
            covered_area = np.pi * self.settings.chaser_detection_radius ** 2

            probability_of_fake_runner_outside_radius = \
                (MAP_AREA - covered_area) / MAP_AREA
            probability_of_fake_runner_inside_radius_but_not_detected = \
                covered_area / MAP_AREA * (1-self.settings.runner_false_positive_probability)

            probability_of_not_detection_one_fake_runner = \
                probability_of_fake_runner_outside_radius + \
                probability_of_fake_runner_inside_radius_but_not_detected

            probability_fake_runner = probability_of_not_detection_one_fake_runner ** self.settings.n_fake_runners

            # for the measure to be None we need to:
            # not measure a real runner
            #    AND
            # not measure a fake runner
            return probability_fake_runner * probability_real_runner

        # the measure has a value (there was a detection)
        probability_real_runner = multivariate_normal.pdf(
            measure.as_numpy(),
            mean= particle.as_numpy(),
            cov= MEASUREMENT_COVARIANCE  #type: ignore
        ) * (1 - self.settings.runner_false_negative_probability)
        
        # fake runner's probability is approximated as a uniform
        # distribution across the map
        probability_of_fake_runner = \
            self.settings.n_fake_runners / MAP_AREA * \
            self.settings.runner_false_positive_probability

        # the final result is:
        # the probability that a real runner generated the measure
        #    PLUS
        # the probability that a fake runner generated the measure
        return probability_real_runner + probability_of_fake_runner

        
    def _normalize(self, points: np.typing.NDArray) -> np.typing.NDArray:
        """
        return the relative coordinated in the range -1, 1
        """
        x = (points[:,0] - MAP_X_LOWER_BOUND) / (MAP_X_UPPER_BOUND - MAP_X_LOWER_BOUND) * 2 - 1
        y = (points[:,1] - MAP_Y_LOWER_BOUND) / (MAP_Y_UPPER_BOUND - MAP_Y_LOWER_BOUND) * 2 - 1
        return np.hstack([x,y])

    def _get_initial_coefficients(self, measure: Point | None, position: Point):
        """
        return the initial coefficients (alpha-hat in n,k) that approximate
        the PDF without relaying on other agent's measures.
        """

        # the log-likelihood associated with each particle
        epsilon = np.log([
            self._probability_of_measure(measure, position, i)
            for i in range(NUMBER_OF_PARTICLES)
        ]).T

        points = self._normalize(self.particles)
        # the evaluation of the Chebyshev polynomials
        # at every point we have particles.
        # shape: [NUM_PARTICLES * NUMBER_OF_APPROXIMATION_COEFFICIENTS]
        theta = chebvander2d(
            points[:,0],
            points[:,1],
            [CHEBYSHEV_ORDER_X, CHEBYSHEV_ORDER_Y]
        )

        # the coefficients for the function approximation
        alpha = np.linalg.pinv(theta).dot(epsilon)


    def run_iteration(self, measure: Point | None):
        pass
        

