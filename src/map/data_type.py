import math

class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    @staticmethod
    def from_polar(angle: float, radius: float) -> Point:
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        return Point(x,y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x,y)


    def __sub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def __str__(self):
        return f"Point{{x={self.x}, y={self.y}}}"

    def abs(self) -> float:
        return (self.x**2 + self.y**2)**0.5


