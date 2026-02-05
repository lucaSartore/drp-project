from abc import ABC, abstractmethod
from map.map import Map
from typing import Self
from map.settings import Settings
import numpy as np

class IController(ABC):

    @classmethod
    @abstractmethod
    def build(
        cls,
        number_of_agents: int,
        agent_id: int,
        settings: Settings
    ) -> Self:
        pass

    @abstractmethod
    def control_loop(self, map: Map):
        pass

    @classmethod
    def subscribe_to_each_other(cls, controllers: list[IController]):
        pass

    @classmethod
    def start_threads(cls, controllers: list[IController]):
        pass


    def get_pdf_image(self) -> np.typing.NDArray | None:
        return None
