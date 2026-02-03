from typing import Literal, Set
from chasers_logic.baseline_controller import BaselineController
from chasers_logic.gaussian_controller import GaussianController
from chasers_logic.icontroller import IController
from chasers_logic.particle_filter_controller import ParticleFilterController
from chasers_logic.pf_manager import ParticleFilterManager
from display.display import Display
from map.map import Map, Settings
from time import time, sleep

def main():
    run_test("baseline")
    return
    for c in ["baseline", "particle_filter"]:
        for i in range(10):
            t = run_test(c, False)  #type: ignore
            print(f"controller={c} iteration={i} time_to_catch={t}")

def run_test(
    controller: Literal["baseline", "gaussian", "particle_filter"],
    enable_display: bool = True,
    settings: Settings = Settings()
) -> int:

    UPDATE_PERIOD = 0.01 #s
    DISPLAY_INTERVALS = 5
    CONTROL_INTERVALS = 2

    map = Map(settings)
    display = Display(settings)
    controller_class: type [IController]
    if controller == "baseline":
        controller_class = BaselineController
    elif controller == "gaussian":
        controller_class = GaussianController
    elif controller == "particle_filter":
        controller_class = ParticleFilterController
    else:
        raise Exception(f"invalid argument: {controller}")

    controllers: list[IController] = [
        controller_class.build(settings.n_chasers, i, settings)
        for i in range(settings.n_chasers)
    ]

    controller_class.subscribe_to_each_other(controllers)
    controller_class.start_threads(controllers)

    counter = 0
    while map.run():
        t1 = time()

        counter += 1

        if counter % CONTROL_INTERVALS == 0:
            for c in controllers:
                c.control_loop(map)

        if counter % DISPLAY_INTERVALS and enable_display:
            map.draw_agents(display)
            c0 = controllers[0]
            if type(c0) == ParticleFilterController:
                pdf = c0.get_pdf_image()
                display.update_right_side(pdf)
            display.render()
            display.clear()

        time_to_sleep = UPDATE_PERIOD - (time()-t1)
        if time_to_sleep > 0 and enable_display:
            sleep(time_to_sleep)

    return counter


if __name__ == '__main__':
    main()
