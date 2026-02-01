from random import Random
from typing import Literal
import numpy as np
from display.display import Display
from map.constants import *
from map.data_type import Point
from map.settings import Settings

class Map:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self._random = Random(settings.random_seed)
        """
        random module (for reproducibility)
        """

        self.runners_positions: np.typing.NDArray[np.float32] = \
            np.zeros((settings.n_fake_runners+1, 2), dtype=np.float32);
        """
        x,y position of the runners (shape [n_runners, 2])
        index zero is the real runner, all of the others are fake one
        """

        self.chasers_positions: np.typing.NDArray[np.float32] = \
            np.zeros((settings.n_chasers, 2), dtype=np.float32);
        """
        x,y position of the chasers (shape [n_chasers, 2])
        index zero is the real runner, all of the others are fake one
        """

        self.chasers = [Chaser(x, self) for x in range(settings.n_chasers)]
        self.runner = Runner(self)
        self.fake_runners = [FakeRunner(x+1, self) for x in range(settings.n_fake_runners)]

    def random(self, lb: float, ub: float) -> float:
        return self._random.random() * (ub - lb) + lb

    def random_angle(self) -> float:
        return self.random(0, 2*np.pi)

    def random_position(self) -> Point:
        x = self.random(MAP_X_LOWER_BOUND, MAP_Y_UPPER_BOUND)
        y = self.random(MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND)
        return Point(x,y)

    def random_coin_flip(self, p_success: float = 0.5) -> bool:
        return self._random.random() <= p_success

    def run(self) -> bool:
        self.runner.step()
        for c in self.chasers:
            c.step()
        for fr in self.fake_runners:
            fr.step()
        return True

    def draw_agents(self, display: Display):
        display.update_left_side(
            self.settings,
            [x.position for x in self.chasers],
            self.runner.position,
            [x.position for x in self.fake_runners]
        )

    @staticmethod
    def _distance_vector(
        origin: Point,
        source: np.typing.NDArray[np.float32]
    ) -> np.typing.NDArray[np.float32]:
        dx = source[:,0] - origin.x
        dy  = source[:,1] - origin.y
        return np.sqrt(dx**2 + dy**2)

    def detect_runner(self, detection_point: Point) -> Point | None:
        distances = self._distance_vector(detection_point, self.runners_positions.copy())
        close_enough =  distances <= self.settings.runner_detection_radius
        for r_index in np.where(close_enough)[0]:
            # detection is real runner
            if r_index == 0 and not self.random_coin_flip(self.settings.runner_false_negative_probability):
                p = self.runner.position
                return self.add_uncertainty(p)
            # detection is fake runner
            if r_index != 0 and self.random_coin_flip(self.settings.runner_false_positive_probability):
                p = self.fake_runners[r_index-1].position
                return self.add_uncertainty(p)
        return None

    def detect_chaser(self, detection_point: Point) -> Point | None:
        distances = self._distance_vector(detection_point, self.chasers_positions)
        close_enough =  distances <= self.settings.chaser_detection_radius
        for c_index in np.where(close_enough)[0]:
            if not self.random_coin_flip(self.settings.chaser_false_negative_probability):
                p = self.chasers[c_index].position
                return self.add_uncertainty(p)
        return None

    def add_uncertainty(self, measurement: Point) -> Point:
        v = np.random.multivariate_normal(
            [measurement.x, measurement.y],
            MEASUREMENT_COVARIANCE
        )
        return Point(v[0], v[1])

class Agent:
    def __init__(
        self,
        map: Map,
        array: np.typing.NDArray[np.float32],
        index: int
    ) -> None:
        self.map = map
        self._array = array
        self._index = index
        self.position = map.random_position()

    @property
    def position(self) -> Point:
        [x,y] = self._array[self._index]
        return Point(x,y)

    @position.setter
    def position(self, var: Point):
        self._array[self._index, :] = [var.x, var.y]

    def reflection(self, surface_normal: float | None, speed: Point) -> Point:
        if surface_normal == None:
            return speed
        if abs(surface_normal - speed.angle) < np.pi/2:
            return speed
        new_angle = np.pi + 2*surface_normal - speed.angle
        new_angle += self.map.random(-BOUNCING_RANDOMNESS, BOUNCING_RANDOMNESS)
        return Point.from_polar(new_angle, speed.module)


class BouncingAgent(Agent):
    def __init__(
        self,
        map: Map,
        array: np.typing.NDArray[np.float32],
        index: int,
        velocity: float
    ) -> None:
        super().__init__(map, array, index)
        self.speed = Point.from_polar(
            map.random_angle(),
            velocity
        )

    def get_bouncing_surface_normal(self) -> float | None:
        surface_normal: float | None = None
        if (self.position.x > MAP_X_UPPER_BOUND):
            surface_normal = -np.pi
        if (self.position.x < MAP_X_LOWER_BOUND):
            surface_normal = 0
        if (self.position.y > MAP_Y_UPPER_BOUND):
            surface_normal = -np.pi/2
        if (self.position.y < MAP_Y_LOWER_BOUND):
            surface_normal = np.pi/2

        if surface_normal != None:
            return surface_normal

        chaser_position = self.map.detect_chaser(self.position)

        if chaser_position == None:
            return None
        
        return (self.position - chaser_position).angle

    def step(self) -> None:
        self.position = self.position + self.speed
        surface_normal = self.get_bouncing_surface_normal()
        self.speed = self.reflection(surface_normal, self.speed)

class Chaser(Agent):
    def __init__(self, index: int, map: Map) -> None:
        super().__init__(map, map.chasers_positions, index)
        self.objective = self.position

    def step(self) -> None:
        direction = self.objective - self.position
        module = direction.module
        if module == 0:
            return
        if module > CHASER_VELOCITY:
            scaling_factor = CHASER_VELOCITY/module
            direction.x *= scaling_factor
            direction.y *= scaling_factor

        self.position = self.position + direction

class Runner(BouncingAgent):
    def __init__(self, map: Map) -> None:
        super().__init__(map, map.runners_positions, 0, RUNNER_VELOCITY)

    def step(self) -> None:
        self.position = self.position + self.speed
        surface_normal = self.get_bouncing_surface_normal()
        self.speed = self.reflection(surface_normal, self.speed)

class FakeRunner(BouncingAgent):
    def __init__(self, index: int, map: Map) -> None:
        super().__init__(map, map.runners_positions, index, RUNNER_VELOCITY)
