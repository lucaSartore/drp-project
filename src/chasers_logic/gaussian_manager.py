from functools import reduce
from typing import Callable, Literal, overload
import numpy as np
from scipy.optimize._lsq.common import print_iteration_linear
from chasers_logic.messages import CoefficientMessage, DistributedKalmanFilterMessage, MeasurementMessage
from map import settings
from map.constants import MAP_AREA, MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND, MEASUREMENT_COVARIANCE, RUNNER_VELOCITY, PARTICLE_UPDATE_COVARIANCE
from chasers_logic.constants import CONSENSUS_ITERATIONS, EXTRA_PARTICLE_BORDER_DISTANCE, NUM_EXTRA_PARTICLES_PER_SIDE, NUMBER_OF_PARTICLES, CHEBYSHEV_ORDER_X, CHEBYSHEV_ORDER_Y, DEBUG, NUMBER_OF_PARTICLES_TO_RESAMPLE_RANDOMLY, WEIGHTS_TO_ASSIGN_TO_RANDOMLY_SAMPLED_PARTICLES
from map.data_type import Point
from map.map import Settings
from scipy.stats import false_discovery_control, multivariate_normal
from numpy.polynomial.chebyshev import chebvander2d
import matplotlib.pyplot as plt
from queue import Queue
from threading import Lock
import copy
from scipy.stats import multivariate_normal


class GaussianManager:
    def __init__(
        self,
        number_of_agents: int,
        agent_id: int,
        settings: Settings
    ) -> None:
        self.number_of_agents = number_of_agents
        self.agent_id = agent_id

        self.messages_queue: Queue[DistributedKalmanFilterMessage] = Queue()
        self.measurement_queue: Queue[MeasurementMessage] = Queue(1)


        self.x_hat = np.zeros(shape=(2,), dtype=np.float32)
        """
        estimated position (x hat) of the kalman filter
        """
        self.P = np.asarray([[25,0],
                             [0,25]], dtype=np.float32)
        """
        error covariance matrix of the current estimate
        """

        self.A = np.asarray([[1,0],
                             [0,1]], dtype=np.float32)
        """
        state transition matrix of the kalman filter
        (in this case it is a identity matrix, as we don't have
         the direction in the state)
        """

        self.Q = np.asarray([[100*RUNNER_VELOCITY**2,0],
                             [0,100*RUNNER_VELOCITY**2]], dtype=np.float32)
        """
        The process noise covariance
        in this case the runner move with a standard deviation equal to
        his velocity
        """

        self.H = np.asarray([[1,0],
                             [0,1]], dtype=np.float32)
        """
        Observation matrix
        (coordinates are 1:1 so this is an identity matrix)
        """

        self.R = MEASUREMENT_COVARIANCE * 100
        """
        Measure noise covariance
        represent the uncertainty on our measures
        """

        self.settings = settings

        self.subscribers: list[Callable[[DistributedKalmanFilterMessage],None]] = []

        self.output_position = Point(0,0)
        self.output_covariance = np.copy(self.P)
        self.output_lock = Lock()

    
    def push_measure(self, message: MeasurementMessage):
        self.measurement_queue.put(message)

    def run(self):
        while True:
            message = self.measurement_queue.get()

            if message.terminal:
                return

            self._run_iteration(message.measurement, message.position)


    def _read_n_messages(self, n: int, iteration: int) -> list[DistributedKalmanFilterMessage]:
        to_return: list[DistributedKalmanFilterMessage] = []
        for _ in range(n):
            message = self.messages_queue.get()
            assert message.agent_id != self.agent_id
            assert message.iteration == iteration
            to_return.append(message)
        return to_return


    def _run_iteration(self, measure: Point | None, position: Point):
        ################## prediction step #######################
        # next state position (prior to the observation)
        x_hat_next_prior = self.A@self.x_hat
        # next state covariance (prior to the observation)
        p_next_prior = self.A@self.P@(self.A.T) + self.Q

        ################## communication step #######################
        if measure == None:
            # trick: if the measure is None we create a fake measure with a super high covariance
            z = Point(0,0).as_numpy()
            R = self.R * 1_000_000
        else:
            z = measure.as_numpy()
            R = self.R
        
        a = self.H.T @ np.linalg.inv(R) @ z
        F = self.H.T @ np.linalg.inv(R) @ self.H

        for i in range(CONSENSUS_ITERATIONS):
            self._send_message_to_subscribers( DistributedKalmanFilterMessage(
                self.agent_id,
                i,
                a,
                F
            ))

            messages = self._read_n_messages(self.number_of_agents-1, i)

            if len(messages) == 0:
                continue

            # here we are making the mesh assumption (all agents are connected to all other agents)
            a = (a + reduce(lambda x,y: x+y, [x.a for x in messages])) / self.number_of_agents
            F = (F + reduce(lambda x,y: x+y, [x.F for x in messages])) / self.number_of_agents
            
        ##################    update step     #######################

        F = F * self.number_of_agents * (1-self.settings.runner_false_positive_probability)
        a = a * self.number_of_agents * (1-self.settings.runner_false_positive_probability)

        self.P = np.linalg.inv(np.linalg.inv(p_next_prior) + F)
        self.x_hat = self.P @ (
            np.linalg.inv(p_next_prior) \
            @ x_hat_next_prior + a
        )

        ################## exporting outputs #######################

        with self.output_lock:
            self.output_position = Point(self.x_hat[0], self.x_hat[1])
            self.output_covariance = np.copy(self.P)

        if DEBUG and self.agent_id == 0:
            self.visualize_pdf()
    
    def _add_to_incoming_messages(self, message: DistributedKalmanFilterMessage):
        self.messages_queue.put(message)

    def _send_message_to_subscribers(self, message: DistributedKalmanFilterMessage):
        for s in self.subscribers:
            s(message)
        
    def subscribe_to(self, other: GaussianManager):
        other.subscribers.append(self._add_to_incoming_messages)

    def read_output(self):
        with self.output_lock:
            pos = copy.copy(self.output_position)
            var = np.copy(self.output_covariance)
            return pos, var



    def visualize_pdf(self, show: bool = True):
        """
        Draws the PDF using a weighted 2D histogram.
        If show=True, displays the plot. If False, returns the image as an RGB array.
        """
        res = 50
        x_bins = np.linspace(MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, res)
        y_bins = np.linspace(MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND, res)

        pos, cov = self.read_output()

        X, Y = np.meshgrid(x_bins, y_bins)
        pos_vec = np.dstack((X, Y))
        statistic = multivariate_normal([pos.x, pos.y], cov).pdf(pos_vec) #type: ignore

        if show:
            fig = plt.figure(figsize=(8, 6))
        else:
            # Create a squared figure with no margins for GUI display
            fig = plt.figure(figsize=(8, 8))

            
        im = plt.imshow(
            statistic, 
            extent=[MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND], #type: ignore
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        
        plt.gca().invert_yaxis()

        if show:
            plt.colorbar(im, label='Covariance matrix')
            plt.title(f"Agent {self.agent_id} covariance matrix")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.show()
        else:
            # Remove all axes and labels for GUI display
            # plt.axis('off')
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
