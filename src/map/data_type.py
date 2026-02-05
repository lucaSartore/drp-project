import math
import numpy as np

class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    @property
    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    @property
    def module(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    @staticmethod
    def from_polar(angle: float, radius: float) -> Point:
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        return Point(x,y)


    def __add__(self, other: Point):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x,y)


    def __sub__(self, other: Point):
        x = self.x - other.x
        y = self.y - other.y
        return Point(x,y)

    def __truediv__(self, n: float | int):
        x = self.x / n
        y = self.y / n
        return Point(x,y)

    def __str__(self):
        return f"Point{{x={self.x}, y={self.y}}}"

    def abs(self) -> float:
        return (self.x**2 + self.y**2)**0.5

    def as_numpy(self) -> np.typing.NDArray[np.float32]:
        return np.array([self.x, self.y], dtype=np.float32)


