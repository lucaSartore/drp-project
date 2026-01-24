from random import Random, random
import numpy as np
from map.constants import *
from map.data_type import Point

class Map:
    def __init__(self, n_chasers: int, n_fake_runners: int, random: Random) -> None:
        self.n_chasers = n_chasers
        """
        number of chasers in the experiment
        """

        self.n_fake_runners = n_fake_runners
        """
        number of fake runners in the experiment
        """

        self._random = random
        """
        random module (for reproducibility)
        """

        self.runners_positions: np.typing.NDArray[np.float32] = \
            np.zeros((n_fake_runners+1, 2), dtype=np.float32);
        """
        x,y position of the runners (shape [n_runners, 2])
        index zero is the real runner, all of the others are fake one
        """

        self.chasers_positions: np.typing.NDArray[np.float32] = \
            np.zeros((n_chasers, 2), dtype=np.float32);
        """
        x,y position of the chasers (shape [n_chasers, 2])
        index zero is the real runner, all of the others are fake one
        """

        self.chasers = [Chaser(x, self) for x in range(n_chasers)]
        self.runner = Runner(self)
        self.fake_runners = [FakeRunner(x+1, self) for x in range(n_fake_runners)]

    def random(self, lb: float, ub: float) -> float:
        return self._random.random() * (ub - lb) + lb

    def random_angle(self) -> float:
        return self.random(0, 2*np.pi)

    def random_position(self) -> Point:
        x = self.random(MAP_X_LOWER_BOUND, MAP_X_LOWER_BOUND)
        y = self.random(MAP_Y_LOWER_BOUND, MAP_Y_LOWER_BOUND)
        return Point(x,y)

    def random_coin_flip(self, p_success: float = 0.5) -> bool:
        return self._random.random() <= p_success

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

class Runner(Agent):
    def __init__(self, map: Map) -> None:
        super().__init__(map.runners_positions, 0)
        self.map = map
        self.orientation = map.random_angle();
        self.position = map.random_position()

class FakeRunner(Agent):
    def __init__(self, index: int, map: Map) -> None:
        super().__init__(map.runners_positions, index)
        self.map = map
        self.position = map.random_position()
        self.orientation = map.random_angle();
