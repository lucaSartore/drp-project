from random import Random
import numpy as np
from map.constants import *
from map.data_type import Point
from dataclasses import dataclass

@dataclass
class Settings:
    n_chasers: int = 3
    n_fake_runners: int = 5
    random_seed: int | None = None
    chaser_false_negative_probability: float = 0.5
    chaser_false_positive_probability: float = 0.5
    chaser_detection_radius: float = 1.5
    runner_false_negative_probability: float = 0.5
    runner_detection_radius: float = 1.5


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

class Agent:
    def __init__(
        self,
        array: np.typing.NDArray[np.float32],
        index: int
    ) -> None:
        self._array = array
        self._index = index

    @property
    def position(self) -> Point:
        [x,y] = self._array[self._index]
        return Point(x,y)

    @position.setter
    def position(self, var: Point):
        self._array[self._index, :] = [var.x, var.y]

class Chaser(Agent):
    def __init__(self, index: int, map: Map) -> None:
        super().__init__(map.chasers_positions, index)
        self.map = map
        self.position = map.random_position()

    def step(self) -> None:
        pass

class Runner(Agent):
    def __init__(self, map: Map) -> None:
        super().__init__(map.runners_positions, 0)
        self.map = map
        self.speed = Point.from_polar(
            map.random_angle(),
            RUNNER_VELOCITY
        )
        self.position = map.random_position()

    def step(self) -> None:
        new_position = self.position + self.speed
        print(self.position, new_position)
        self.position = new_position
        if (new_position.x > MAP_X_UPPER_BOUND and self.speed.x > 0):
            self.speed.x *= -1
        if (new_position.x < MAP_X_LOWER_BOUND and self.speed.x < 0):
            self.speed.x *= -1
        if (new_position.y > MAP_Y_UPPER_BOUND and self.speed.y > 0):
            self.speed.y *= -1
        if (new_position.y < MAP_Y_LOWER_BOUND and self.speed.y < 0):
            self.speed.y *= -1

class FakeRunner(Agent):
    def __init__(self, index: int, map: Map) -> None:
        super().__init__(map.runners_positions, index)
        self.map = map
        self.position = map.random_position()
        self.orientation = map.random_angle();
    def step(self) -> None:
        pass
