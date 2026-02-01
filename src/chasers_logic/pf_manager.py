from itertools import product
from typing import Callable, Literal, overload, override
import numpy as np
from chasers_logic.messages import CoefficientMessage, MeasurementMessage
from map.constants import MAP_AREA, MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND, MEASUREMENT_COVARIANCE, RUNNER_VELOCITY
from chasers_logic.constants import CONSENSUS_ITERATIONS, NUMBER_OF_PARTICLES, CHEBYSHEV_ORDER_X, CHEBYSHEV_ORDER_Y, DEBUG
from map.data_type import Point
from map.map import Settings
from scipy.stats import multivariate_normal
from numpy.polynomial.chebyshev import chebvander2d
import matplotlib.pyplot as plt
from queue import Queue
from scipy.stats import gaussian_kde


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

        self.coefficients_queue: Queue[CoefficientMessage] = Queue()
        self.measurement_queue: Queue[MeasurementMessage] = Queue(1)

        self.settings = settings

        self.subscribers: list[Callable[[CoefficientMessage],None]] = []


    def _get_particle(self, index: int) -> Point:
        return Point(
            self.particles[index,0],
            self.particles[index,1]
        )

    def get_random_particle(self) -> Point:
        index = np.random.choice(len(self.weights), p=self.weights)
        return self._get_particle(index)

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

    def _chebvander_weighted(self, points: np.typing.NDArray, weights: np.typing.NDArray):
        chebvander = self._chebvander(points)
        results = chebvander @ weights
        return results

    def _chebvander(self, points: np.typing.NDArray):
        points = self._normalize(points)
        return chebvander2d(
            points[:,0],
            points[:,1],
            [CHEBYSHEV_ORDER_X, CHEBYSHEV_ORDER_Y]
        )

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

        # the evaluation of the Chebyshev polynomials
        # at every point we have particles.
        # shape: [NUM_PARTICLES, NUMBER_OF_APPROXIMATION_COEFFICIENTS]
        theta = self._chebvander(self.particles)

        # the coefficients for the function approximation
        # shape: [NUMBER_OF_APPROXIMATION_COEFFICIENTS]
        alpha = np.linalg.pinv(theta)@(epsilon)

        return alpha

    @overload
    def visualize_approximation_coefficients(self, alpha: np.typing.NDArray, show: Literal[True] = True):
        pass
    @overload
    def visualize_approximation_coefficients(self, alpha: np.typing.NDArray, show: Literal[False]) -> np.typing.NDArray:
        pass
    def visualize_approximation_coefficients(self, alpha: np.typing.NDArray, show: bool = True):
        """
        Visualizes the approximated log-likelihood field.
        If show=True, displays the plot. If False, returns the image as an RGB array.
        """
        res = 100
        x_range = np.linspace(MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, res)
        y_range = np.linspace(MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND, res)
        X, Y = np.meshgrid(x_range, y_range)

        points_to_eval = np.vstack([X.ravel(), Y.ravel()]).T
        Z = self._chebvander_weighted(points_to_eval, alpha)
        Z = Z.reshape(res, res)
        
        fig = plt.figure(figsize=(8, 6))
        im = plt.imshow(
            np.exp(Z), 
            extent=[MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND], #type: ignore
            origin='lower',
            cmap='viridis'
        )
        
        plt.scatter(self.particles[:, 0], self.particles[:, 1], s=1, c='red', alpha=0.5, label='Particles')
        plt.colorbar(im, label='Approximated Probability Density')
        plt.title(f"Agent {self.agent_id} Probability Distribution Approximation")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.gca().invert_yaxis()

        if show:
            plt.show()
        else:
            return self._canvas_to_array(fig)

    @overload
    def visualize_pdf(self, show: Literal[True] = True) -> None:
        pass
    @overload
    def visualize_pdf(self, show: Literal[False]) -> np.typing.NDArray:
        pass
    def visualize_pdf(self, show: bool = True):
        """
        Draws the PDF using a weighted 2D histogram.
        If show=True, displays the plot. If False, returns the image as an RGB array.
        """
        res = 50
        x_bins = np.linspace(MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, res)
        y_bins = np.linspace(MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND, res)

        statistic, _, _ = np.histogram2d(
            self.particles[:, 0], 
            self.particles[:, 1], 
            bins=[x_bins, y_bins], 
            weights=self.weights,
            density=True
        )

        if show:
            fig = plt.figure(figsize=(8, 6))
        else:
            # Create a squared figure with no margins for GUI display
            fig = plt.figure(figsize=(8, 8))
            
        im = plt.imshow(
            statistic.T, 
            extent=[MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND], #type: ignore
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        
        plt.scatter(self.particles[:, 0], self.particles[:, 1], s=0.5, c='red', alpha=0.2)
        plt.gca().invert_yaxis()

        if show:
            plt.colorbar(im, label='Weight Density')
            plt.title(f"Agent {self.agent_id} Raw Particle Density (Weighted Histogram)")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.show()
        else:
            # Remove all axes and labels for GUI display
            plt.axis('off')
            # Remove all margins and padding
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            return self._canvas_to_array(fig)

    def _canvas_to_array(self, fig) -> np.typing.NDArray:
        """Helper to convert a matplotlib figure to a RGB numpy array."""
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # removing alpha channel
        img = img[:,:,1:]
        plt.close(fig) # Clean up memory
        return img

    def push_measure(self, message: MeasurementMessage):
        self.measurement_queue.put(message)

    def run(self):
        while True:
            message = self.measurement_queue.get()

            if message.terminal:
                return

            self._run_iteration(message.measurement, message.position)

            if self.agent_id == 0 and DEBUG:
                self.visualize_pdf()
            


    def _read_n_messages(self, n: int, iteration: int) -> list[CoefficientMessage]:
        to_return: list[CoefficientMessage] = []
        for _ in range(n):
            message = self.coefficients_queue.get()
            assert message.iteration == iteration
            assert message.agent_id != self.agent_id
            to_return.append(message)
        return to_return


    def _run_iteration(self, measure: Point | None, position: Point):
        self._update_particles()

        alpha = self._get_initial_coefficients(measure, position)

        # parameters to estimate
        zeta_current = alpha.copy()
        zeta_next = np.zeros_like(zeta_current)

        for i in range(CONSENSUS_ITERATIONS):

            self._send_message_to_subscribers(CoefficientMessage(
                self.agent_id,
                i,
                zeta_current
            ))

            # parameters
            messages = self._read_n_messages(self.settings.n_chasers-1, i)

            zeta_next = zeta_current + np.sum([x.coefficients for x in messages], axis = 0)
            zeta_next /= self.settings.n_chasers

            zeta_current = zeta_next

        if self.agent_id == 0  and DEBUG:
            self.visualize_approximation_coefficients(zeta_next)

        # update probability based on measures
        probability = np.exp(self._chebvander_weighted(self.particles, zeta_next))

        self.weights *= probability

        # normalization
        assert np.sum(self.weights) != 0
        self.weights /= np.sum(self.weights)

        # redistribute probability to avoid getting to much confidence
        # self.weights += 1 / NUMBER_OF_PARTICLES
        # self.weights /= np.sum(self.weights)


    def _add_to_incoming_messages(self, message: CoefficientMessage):
        self.coefficients_queue.put(message)

    def _send_message_to_subscribers(self, message: CoefficientMessage):
        for s in self.subscribers:
            s(message)
        
    def subscribe_to(self, other: ParticleFilterManager):
        other.subscribers.append(self._add_to_incoming_messages)

    def _update_particles(self):
        # add a random velocity
        angle = np.random.random(size = (NUMBER_OF_PARTICLES)) * np.pi * 2
        p = self.particles
        p[:,0] += RUNNER_VELOCITY * np.cos(angle)
        p[:,1] += RUNNER_VELOCITY * np.sin(angle)

        # keep particles in the border
        x = p[:,0]
        y = p[:,1]
        x[x < MAP_X_LOWER_BOUND] = 2 * MAP_X_LOWER_BOUND - x[x < MAP_X_LOWER_BOUND]
        x[x > MAP_X_UPPER_BOUND] = 2 * MAP_X_UPPER_BOUND - x[x > MAP_X_UPPER_BOUND]
        y[y < MAP_Y_LOWER_BOUND] = 2 * MAP_Y_LOWER_BOUND - y[y < MAP_Y_LOWER_BOUND]
        y[y > MAP_Y_UPPER_BOUND] = 2 * MAP_Y_UPPER_BOUND - y[y > MAP_Y_UPPER_BOUND]
        p[:,0] = x
        p[:,1] = y
