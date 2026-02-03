from chasers_logic.icontroller import IController
from map.data_type import Point
from map.map import Map
from map.settings import Settings
from typing import Self


class GaussianController(IController):

    NEW_OBJ_TH = 0.1
    def __init__(
        self,
        agent_id: int
    ) -> None:
        self.agent_id = agent_id
        self.objective = Point(0,0)

    @classmethod
    def build(cls, number_of_agents: int, agent_id: int, settings: Settings) -> Self:
        return cls(agent_id)

    def control_loop(self, map: Map):
        position = map.chasers[self.agent_id].position
        measure = map.detect_runner(position)

        # if we have a measure we go towards it
        if measure != None:
            self.objective = measure
        # if we reached the objective we sample a new one
        elif (self.objective - position).abs() <= GaussianController.NEW_OBJ_TH:
            self.objective = map.random_position()
        
        map.chasers[self.agent_id].objective = self.objective

