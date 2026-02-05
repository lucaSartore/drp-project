from math import dist
from queue import Queue
from threading import Thread
import sys
from chasers_logic.gaussian_manager import GaussianManager
from chasers_logic.icontroller import IController
from map.constants import MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND
from map.map import Settings, Map
from map.data_type import Point
from chasers_logic.pf_manager import ParticleFilterManager
from chasers_logic.messages import MeasurementMessage
from itertools import product
import numpy as np
from typing import Self
from scipy.stats import multivariate_normal

class GaussianController(IController):
    VAR_THRESHOLD_FOR_CHASING = 1.0
    SEARCH_LOOP_DISENGAGEMENT_THRESHOLD = 0.5

    def __init__(
        self,
        number_of_agents: int,
        agent_id: int,
        settings: Settings
    ) -> None:
        self.agent_id = agent_id
        self.number_of_agents = 0
        self.gm = GaussianManager(number_of_agents, agent_id, settings)
        self.probability_of_objective = 1.0
        self.objective = Point(0,0)


    @classmethod
    def build(cls, number_of_agents: int, agent_id: int, settings: Settings) -> Self:
        return cls(number_of_agents, agent_id, settings)


    def get_pdf_image(self):
        return self.gm.visualize_pdf(False)

    def control_loop(self, map: Map):
        position = map.chasers[self.agent_id].position
        measure = map.detect_runner(position)
        self.gm.push_measure(MeasurementMessage(
            measure,
            position
        ))

        pos, cov = self.gm.read_output()

        if self._is_chasing_mode(cov):
            result = self._chasing_mode_control_loop(pos)
        else:
            result = self._search_mode_control_loop(pos, cov, position)

        map.chasers[self.agent_id].objective = result

    def _chasing_mode_control_loop(self, pos) -> Point:
        """
        chasing mode is activated when the robot has a clear idea on where the 
        runner is, and is simply trying to follow him
        """
        return pos

    def _search_mode_control_loop(self, pos: Point, cov: np.typing.NDArray, position: Point) -> Point:
        """
        search mode is activated when the robot does not know the exact position 
        of the runner, and simply move towards regions where the probability of
        finding the agents is high
        """
        prob_in_target = multivariate_normal.pdf(self.objective.as_numpy(), mean=pos.as_numpy(), cov=cov) #type: ignore

        disengage_because_of_probability = prob_in_target < GaussianController.SEARCH_LOOP_DISENGAGEMENT_THRESHOLD * self.probability_of_objective
        disengage_because_of_reached_point = (position - self.objective).abs() < 0.1
        if disengage_because_of_probability or disengage_because_of_reached_point:
            return self._search_mode_disengagement(pos, cov)
        else:
            return self._search_mode_continuation(pos, cov)

    def _search_mode_disengagement(self, pos: Point, cov: np.typing.NDArray) -> Point:
        new_target = np.random.multivariate_normal(pos.as_numpy(), cov, (1,))
        print(new_target)
        prob_in_target = multivariate_normal.pdf(new_target, mean=pos.as_numpy(), cov=cov) #type: ignore

        new_target_point = Point(new_target[0,0], new_target[0,1])
        new_target_point.x = np.clip(new_target_point.x,MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND)
        new_target_point.y = np.clip(new_target_point.y,MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND)
        self.objective = new_target_point
        self.probability_of_objective = prob_in_target
        return new_target_point


    def _search_mode_continuation(self, pos: Point, cov: np.typing.NDArray) -> Point:
        self.objective = pos
        return pos

    def _is_chasing_mode(self, cov: np.typing.NDArray) -> bool:
        return cov[0,0] < GaussianController.VAR_THRESHOLD_FOR_CHASING and \
            cov[1,1] < GaussianController.VAR_THRESHOLD_FOR_CHASING

    def _distances(self, particles: np.typing.NDArray, center: Point | np.typing.NDArray) -> np.typing.NDArray:
        if type(center) == Point:
            center = center.as_numpy()
        diff = particles - center
        distance = np.sqrt(np.sum(diff ** 2, axis=1))
        return distance

    @classmethod
    def subscribe_to_each_other(cls, controllers: list[IController]):
        for (a,b) in product(controllers, controllers):
            assert type(a) == GaussianController
            assert type(b) == GaussianController
            if a.gm.agent_id != b.gm.agent_id:
                a.gm.subscribe_to(b.gm)

    @classmethod
    def start_threads(cls, controllers: list[IController]):
        for c in controllers:
            assert type(c) == GaussianController
            Thread(target= c.gm.run).start()


