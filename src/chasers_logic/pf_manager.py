import numpy as np
from map.constants import MAP_AREA, MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND, MEASUREMENT_COVARIANCE
from chasers_logic.constants import NUMBER_OF_PARTICLES, CHEBYSHEV_ORDER_X, CHEBYSHEV_ORDER_Y
from map.data_type import Point
from map.map import Settings
from scipy.stats import multivariate_normal
from numpy.polynomial.chebyshev import chebvander2d
import matplotlib.pyplot as plt


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
        """
        calculate the probability of a certain measure to be made
        given a certain particle
        """
        particle = self._get_particle(particle_index)

        # the measure is null (no detection)
        if measure == None:
            is_in_radius = (position-particle).abs() <= self.settings.chaser_detection_radius 
            probability_of_measure_given_runner = 1 if not is_in_radius else self.settings.runner_false_negative_probability


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

            probability_of_measure_given_fake_runner = probability_of_not_detection_one_fake_runner ** self.settings.n_fake_runners

            # for the measure to be None we need to:
            # not measure a real runner
            #    AND
            # not measure a fake runner
            return probability_of_measure_given_fake_runner * probability_of_measure_given_runner

        # the measure has a value (there was a detection)
        probability_of_real_runner_detection = multivariate_normal.pdf(
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
        return probability_of_real_runner_detection + probability_of_fake_runner

        
    def _normalize(self, points: np.typing.NDArray) -> np.typing.NDArray:
        """
        return the relative coordinated in the range -1, 1
        """
        x = (points[:,0] - MAP_X_LOWER_BOUND) / (MAP_X_UPPER_BOUND - MAP_X_LOWER_BOUND) * 2 - 1
        y = (points[:,1] - MAP_Y_LOWER_BOUND) / (MAP_Y_UPPER_BOUND - MAP_Y_LOWER_BOUND) * 2 - 1
        return np.vstack([x,y]).T

    def _get_initial_coefficients(self, measure: Point | None, position: Point) -> np.typing.NDArray:
        """
        return the initial coefficients (alpha-hat in n,k) that approximate
        the PDF without relaying on other agent's measures.
        """

        # the log-likelihood associated with each particle
        epsilon = [
            self._probability_of_measure(measure, position, i)
            for i in range(NUMBER_OF_PARTICLES)
        ]
        epsilon = np.log(epsilon).T

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
        # shape: [NUMBER_OF_APPROXIMATION_COEFFICIENTS]
        x = np.linalg.pinv(theta)
        x = np.isnan(x)
        count = np.count_nonzero(x)
        alpha = np.linalg.pinv(theta)@(epsilon)

        return alpha

    def _visualize_coefficients(self, alpha: np.typing.NDArray):
        """
        Visualizes the approximated log-likelihood field using the 
        Chebyshev coefficients.
        """
        # 1. Create a grid of points in the map coordinate space
        res = 100  # Resolution of the heat map
        x_range = np.linspace(MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, res)
        y_range = np.linspace(MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND, res)
        X, Y = np.meshgrid(x_range, y_range)

        # 2. Normalize the grid points to the [-1, 1] range for Chebyshev evaluation
        points_to_eval = np.vstack([X.ravel(), Y.ravel()]).T
        normalized_points = self._normalize(points_to_eval)

        # 3. Build the Vandermonde matrix for the grid
        theta_grid = chebvander2d(
            normalized_points[:, 0],
            normalized_points[:, 1],
            [CHEBYSHEV_ORDER_X, CHEBYSHEV_ORDER_Y]
        )

        # 4. Evaluate the function (log-likelihood) at each grid point
        # Z = Theta * alpha
        Z = theta_grid @ alpha
        Z = Z.reshape(res, res)

        # 5. Plotting
        plt.figure(figsize=(8, 6))
        
        # Display as a heatmap (exponentiated to show probability if desired)
        # We use origin='lower' to match Cartesian coordinates
        im = plt.imshow(
            np.exp(Z), 
            extent=[MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND],
            origin='lower',
            cmap='viridis'
        )
        
        # Overlay the current particles to see how they align with the approximation
        plt.scatter(self.particles[:, 0], self.particles[:, 1], s=1, c='red', alpha=0.5, label='Particles')
        
        plt.colorbar(im, label='Approximated Probability Density')
        plt.title(f"Agent {self.agent_id} Probability Distribution Approximation")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()


    def run_iteration(self, measure: Point | None, position: Point):
        alpha = self._get_initial_coefficients(measure, position)
        self._visualize_coefficients(alpha)
        

