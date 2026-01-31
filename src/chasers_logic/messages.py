from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass


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
    
