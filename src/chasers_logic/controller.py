from math import dist
from threading import Thread
import sys
from map.map import Settings, Map
from map.data_type import Point
from chasers_logic.pf_manager import ParticleFilterManager
from chasers_logic.messages import MeasurementMessage
from itertools import product
import numpy as np
import random

class ChaserController:
    RADIUS_FOR_SEARCH_LOOP = 3.0
    SEARCH_LOOP_DISENGAGEMENT_THRESHOLD = 0.5
    STD_THRESHOLD_FOR_CHASING = 1.0
    """
    if std of particle is smaller than the constant here
    the robot will enter "chasing mode" where it move towards
    the mean of the particles. otherwise it will remain in "search mode"
    """

    def __init__(
        self,
        number_of_agents: int,
        agent_id: int,
        settings: Settings
    ) -> None:
        self.agent_id = agent_id
        self.pfm = ParticleFilterManager(number_of_agents, agent_id, settings)
        self.objective = Point(0,0)
        self.num_particle_around_objective = sys.maxsize


    def get_pdf_image(self):
        return self.pfm.visualize_pdf(False)

    def control_loop(self, map: Map):
        position = map.chasers[self.agent_id].position
        measure = map.detect_runner(position)
        self.pfm.push_measure(MeasurementMessage(
            measure,
            position
        ))

        particles = self.pfm.read_output()

        if self._is_chasing_mode(particles):
            result = self._chasing_mode_control_loop(particles)
        else:
            result = self._search_mode_control_loop(particles, position)

        map.chasers[self.agent_id].objective = result

    def _chasing_mode_control_loop(self, particles: np.typing.NDArray) -> Point:
        """
        chasing mode is activated when the robot has a clear idea on where the 
        runner is, and is simply trying to follow him
        """
        mean = np.mean(particles,axis=0)
        return Point(mean[0], mean[1])

    def _search_mode_control_loop(self, particles: np.typing.NDArray, position: Point) -> Point:
        """
        search mode is activated when the robot does not know the exact position 
        of the runner, and simply move towards regions where the probability of
        finding the agents is high
        """
        distance = self._distances(particles, self.objective)
        in_radius = distance < ChaserController.RADIUS_FOR_SEARCH_LOOP
        counter = np.count_nonzero(in_radius)

        if counter < ChaserController.SEARCH_LOOP_DISENGAGEMENT_THRESHOLD * self.num_particle_around_objective:
            return self._search_mode_disengagement(particles, position)
        else:
            return self._search_mode_continuation(particles[in_radius])

    def _search_mode_disengagement(self, particles: np.typing.NDArray, position: Point) -> Point:
        # pick a random particles, with selection probability inversely
        # proportional to distance (to avoid zig-zagging in the map)
        distances_from_self = self._distances(particles, position)
        weights = 1/(1+distances_from_self)
        weights /= np.sum(weights)
        index = np.random.choice(len(particles), p=weights)
        # set the particle as the next objective
        objective = Point(particles[index,0], particles[index,1])
        distance = self._distances(particles, objective)
        in_radius = distance < ChaserController.RADIUS_FOR_SEARCH_LOOP
        counter = np.count_nonzero(in_radius)

        self.objective = objective
        self.num_particle_around_objective = counter
        return objective


    def _search_mode_continuation(self, particles_in_radius: np.typing.NDArray) -> Point:
        mean = np.mean(particles_in_radius,axis=0)
        point = Point(mean[0], mean[1])
        self.objective = point
        return point

    def _is_chasing_mode(self, particles: np.typing.NDArray) -> bool:
        mean = np.mean(particles,axis=0)
        distance = self._distances(particles, mean)
        std = float(np.std(distance))
        return std < ChaserController.STD_THRESHOLD_FOR_CHASING

    def _distances(self, particles: np.typing.NDArray, center: Point | np.typing.NDArray) -> np.typing.NDArray:
        if type(center) == Point:
            center = center.as_numpy()
        diff = particles - center
        distance = np.sqrt(np.sum(diff ** 2, axis=1))
        return distance

    @staticmethod
    def subscribe_to_each_other(controllers: list[ChaserController]):
        for (a,b) in product(controllers, controllers):
            if a.pfm.agent_id != b.pfm.agent_id:
                a.pfm.subscribe_to(b.pfm)

    @staticmethod
    def start_threads(controllers: list[ChaserController]):
        for c in controllers:
            Thread(target= c.pfm.run).start()


