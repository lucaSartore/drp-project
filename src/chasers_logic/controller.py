from threading import Thread
from map.map import Settings, Map
from chasers_logic.pf_manager import ParticleFilterManager
from chasers_logic.messages import MeasurementMessage
from itertools import product
import numpy as np

class ChaserController:
    pass

    def __init__(
        self,
        number_of_agents: int,
        agent_id: int,
        settings: Settings
    ) -> None:
        self.agent_id = agent_id
        self.pfm = ParticleFilterManager(number_of_agents, agent_id, settings)


    def get_pdf_image(self):
        return self.pfm.visualize_pdf(False)

    def control_loop(self, map: Map):
        position = map.chasers[self.agent_id].position
        measure = map.detect_runner(position)
        self.pfm.push_measure(MeasurementMessage(
            measure,
            position
        ))

        chaser = map.chasers[self.agent_id]
        obj_distance = chaser.objective - chaser.position
        if obj_distance.module < 0.5:
            map.chasers[self.agent_id].objective = self.pfm.get_random_particle()


    @staticmethod
    def subscribe_to_each_other(controllers: list[ChaserController]):
        for (a,b) in product(controllers, controllers):
            if a.pfm.agent_id != b.pfm.agent_id:
                a.pfm.subscribe_to(b.pfm)

    @staticmethod
    def start_threads(controllers: list[ChaserController]):
        for c in controllers:
            Thread(target= c.pfm.run).start()


