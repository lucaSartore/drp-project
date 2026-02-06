from typing import Callable, Literal, overload
import numpy as np
from chasers_logic.messages import CoefficientMessage, MeasurementMessage
from map.constants import MAP_AREA, MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND, MEASUREMENT_COVARIANCE, RUNNER_VELOCITY, PARTICLE_UPDATE_COVARIANCE
from chasers_logic.constants import CONSENSUS_ITERATIONS, EXTRA_PARTICLE_BORDER_DISTANCE, NUM_EXTRA_PARTICLES_PER_SIDE, NUMBER_OF_PARTICLES, CHEBYSHEV_ORDER_X, CHEBYSHEV_ORDER_Y, DEBUG, NUMBER_OF_PARTICLES_TO_RESAMPLE_RANDOMLY, WEIGHTS_TO_ASSIGN_TO_RANDOMLY_SAMPLED_PARTICLES
from map.data_type import Point
from map.map import Settings
from scipy.stats import multivariate_normal
from numpy.polynomial.chebyshev import chebvander2d
import matplotlib.pyplot as plt
from queue import Queue
from threading import Lock


class ParticleFilterManager:
    def __init__(
        self,
        number_of_agents: int,
        agent_id: int,
        settings: Settings
    ) -> None:
        self.number_of_agents = number_of_agents
        self.agent_id = agent_id
        self.particles = self._get_random_particles(NUMBER_OF_PARTICLES)
        """
        particles in the particle filters
        shape: [NUMBER_OF_PARTICLES, 2] (where 2 is x,y)
        """

        self.weights = np.ones(shape=(NUMBER_OF_PARTICLES), dtype=np.float32) * 1/NUMBER_OF_PARTICLES
        """
        weights associated with each particle
        shape: [NUMBER_OF_PARTICLES]
        """

        self.coefficients_queue: Queue[CoefficientMessage] = Queue()
        self.measurement_queue: Queue[MeasurementMessage] = Queue(1)

        self.settings = settings

        self.subscribers: list[Callable[[CoefficientMessage],None]] = []

        self.output_particles = self.particles.copy()
        self.output_lock = Lock()
        self.extra_particles = self._get_extra_particles()

    def _get_random_particles(self, count: int) -> np.typing.NDArray:
        particles = np.zeros(shape=(count, 2), dtype=np.float32)
        rand = lambda lb, ub: np.random.random(count) * (ub - lb) + lb
        particles[:,0] = rand(MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND)
        particles[:,1] = rand(MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND)
        return particles


    def _get_extra_particles(self):
        """
        return extra particle that are added when finding the coefficients of 
        the chebyshev approximator.
        These are added outside the boundry of the map and they allow the solution
        to be more precise around the edges avoiding the creation of "ghosts"
        """
        range_x = np.linspace(
            MAP_X_LOWER_BOUND,
            MAP_X_UPPER_BOUND,
            NUM_EXTRA_PARTICLES_PER_SIDE
        )
        range_y = np.linspace(
            MAP_Y_LOWER_BOUND,
            MAP_Y_UPPER_BOUND,
            NUM_EXTRA_PARTICLES_PER_SIDE
        )

        template = np.zeros((NUM_EXTRA_PARTICLES_PER_SIDE,2), dtype=np.float32)

        particles_top = np.zeros_like(template)
        particles_top[:,0] = range_x
        particles_top[:,1] = MAP_Y_UPPER_BOUND + EXTRA_PARTICLE_BORDER_DISTANCE

        particles_bottom = np.zeros_like(template)
        particles_bottom[:,0] = range_x
        particles_bottom[:,1] = MAP_Y_LOWER_BOUND - EXTRA_PARTICLE_BORDER_DISTANCE

        particles_left = np.zeros_like(template)
        particles_left[:,1] = range_y
        particles_left[:,0] = MAP_X_LOWER_BOUND - EXTRA_PARTICLE_BORDER_DISTANCE

        particles_right = np.zeros_like(template)
        particles_right[:,1] = range_y
        particles_right[:,0] = MAP_X_UPPER_BOUND + EXTRA_PARTICLE_BORDER_DISTANCE


        return np.vstack([
            particles_top,
            particles_bottom,
            particles_left,
            particles_right
        ])


    def _get_particle(self, index: int) -> Point:
        return Point(
            self.particles[index,0],
            self.particles[index,1]
        )

    def get_random_particle(self) -> Point:
        index = np.random.choice(len(self.weights), p=self.weights)
        return self._get_particle(index)

    def _probability_of_not_detecting_fake_runner(self):
        """
        return an approximation of the probability that in any given time,
        the measure is null given that the real runner did not cause the measure
        """

        if self.settings.n_fake_runners == 0:
            return 1

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

        # the condition need to be true at the same time for all runners
        return probability_of_not_detection_one_fake_runner ** self.settings.n_fake_runners

    def _probability_of_measures(self, measure: Point | None, position: Point, points: np.typing.NDArray) -> np.typing.NDArray:
        """
        calculate the probability of a certain measure to be made
        assuming each of the particle is the source

        more efficient parallel version of `_probability_of_measure`
        """

        ##### case where there is not a detection #####
        if measure == None:
            distance = points - position.as_numpy()
            distance = np.sqrt(np.sum(distance ** 2, axis=1))
            is_in_radius = distance < self.settings.chaser_detection_radius
            # if the runner is outside the radius, the probability of not detection
            # a measure is 100%
            probability = np.ones_like(points[:,0])
            # if the runner is in radius the probability of not seeing him
            # is equal to the false negative probability
            probability[is_in_radius] = self.settings.runner_false_negative_probability

            # keep also track of the probability that a fake runner generates a measure
            probability *= self._probability_of_not_detecting_fake_runner()

            return probability

        ##### case where there is a measure #####
        probability_of_real_runner_detection = multivariate_normal.pdf(
            measure.as_numpy() - points,
            mean= [0,0],
            cov= MEASUREMENT_COVARIANCE  #type: ignore
        ) * (1 - self.settings.runner_false_negative_probability)
        
        # fake runner's probability is approximated as a uniform
        # distribution across the map
        probability_of_fake_runner = \
            self.settings.n_fake_runners / MAP_AREA * \
            self.settings.runner_false_positive_probability

        # enhance numerical stability
        # if I get a measure that no particle is abl to satisfy
        # then all the probability are zero, and will result in
        # coefficients going to infinity
        if probability_of_fake_runner == 0:
            probability_of_fake_runner = 0.001

        # the final result is:
        # the probability that a real runner generated the measure
        #    PLUS
        # the probability that a fake runner generated the measure
        return probability_of_real_runner_detection + probability_of_fake_runner
    


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




            if self.settings.n_fake_runners != 0:
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
            else:
                probability_of_measure_given_fake_runner = 1

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

        # enhance numerical stability
        # if I get a measure that no particle is abl to satisfy
        # then all the probability are zero, and will result in
        # coefficients going to infinity
        if probability_of_fake_runner == 0:
            probability_of_fake_runner = 0.001

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

        samples = np.vstack([self.particles, self.extra_particles])

        # the log-likelihood associated with each particle
        epsilon2 = self._probability_of_measures(measure, position, samples)
        # epsilon = [
        #     self._probability_of_measure(measure, position, i)
        #     for i in range(NUMBER_OF_PARTICLES)
        # ]
        epsilon = np.log(epsilon2).T

        # the evaluation of the Chebyshev polynomials
        # at every point we have particles.
        # shape: [NUM_PARTICLES, NUMBER_OF_APPROXIMATION_COEFFICIENTS]
        theta = self._chebvander(samples)

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

        assert not any(np.isnan(probability))
        # assert not all(probability >= 0)
        # assert not all(probability <= 1)
        probability = np.clip(probability,0,1)
        # print(probability[probability>1])
        # print(probability[probability<0])

        # self.weights *= self._probability_of_measures(measure, position, self.particles)
        self.weights *= probability


        # normalization
        sum = np.sum(self.weights)
        assert sum != 0
        self.weights /= sum


        assert not any(np.isnan(self.weights))

        self._resampling()
        self._write_output()

    def _resampling(self):
        to_resample = NUMBER_OF_PARTICLES - NUMBER_OF_PARTICLES_TO_RESAMPLE_RANDOMLY
        # resampling
        indices = np.random.choice(np.arange(NUMBER_OF_PARTICLES), size=to_resample, p=self.weights)

        resampled_particles = self.particles[indices]
        random_particles = self._get_random_particles(NUMBER_OF_PARTICLES_TO_RESAMPLE_RANDOMLY)

        self.particles = np.vstack([resampled_particles, random_particles])
        self.weights[:to_resample] = (1-WEIGHTS_TO_ASSIGN_TO_RANDOMLY_SAMPLED_PARTICLES)/to_resample
        self.weights[to_resample:] = (WEIGHTS_TO_ASSIGN_TO_RANDOMLY_SAMPLED_PARTICLES)/NUMBER_OF_PARTICLES_TO_RESAMPLE_RANDOMLY

    def _add_to_incoming_messages(self, message: CoefficientMessage):
        self.coefficients_queue.put(message)

    def _send_message_to_subscribers(self, message: CoefficientMessage):
        for s in self.subscribers:
            s(message)
        
    def subscribe_to(self, other: ParticleFilterManager):
        other.subscribers.append(self._add_to_incoming_messages)

    def _update_particles(self):
        # add a random velocity
        displacement = np.random.multivariate_normal(mean=[0,0], cov = PARTICLE_UPDATE_COVARIANCE, size = NUMBER_OF_PARTICLES)
        p = self.particles + displacement

        # keep particles in the border
        x = p[:,0]
        y = p[:,1]
        x[x < MAP_X_LOWER_BOUND] = 2 * MAP_X_LOWER_BOUND - x[x < MAP_X_LOWER_BOUND]
        x[x > MAP_X_UPPER_BOUND] = 2 * MAP_X_UPPER_BOUND - x[x > MAP_X_UPPER_BOUND]
        y[y < MAP_Y_LOWER_BOUND] = 2 * MAP_Y_LOWER_BOUND - y[y < MAP_Y_LOWER_BOUND]
        y[y > MAP_Y_UPPER_BOUND] = 2 * MAP_Y_UPPER_BOUND - y[y > MAP_Y_UPPER_BOUND]
        p[:,0] = x
        p[:,1] = y

        self.particles = p


    def read_output(self):
        with self.output_lock:
            return self.output_particles

    def _write_output(self):
        with self.output_lock:
            to_export = NUMBER_OF_PARTICLES - NUMBER_OF_PARTICLES_TO_RESAMPLE_RANDOMLY
            self.output_particles = self.particles.copy()[:to_export]

