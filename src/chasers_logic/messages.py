from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass

from map.data_type import Point


@dataclass
class CoefficientMessage:
    agent_id: int
    """
    the id of the agent's who have generated this coefficients
    """
    iteration: int
    """
    iteration of the approximation algorithm
    """
    coefficients: NDArray[np.float32]
    """
    coefficients of the agent.
    shape: [num_approx_functions]
    """
    
@dataclass
class MeasurementMessage:
    measurement: Point | None
    """
    measurement taken by the robot
    """
    position: Point
    """
    the position the robot was when he took the measurement
    """
    terminal: bool = False
    """
    condition for terminating
    """
